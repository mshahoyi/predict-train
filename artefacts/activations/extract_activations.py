#!/usr/bin/env python3
"""Extract per-layer activations from datasets using a local LLM."""

# %%
import argparse
import json
import logging
import pickle
import shutil
from pathlib import Path

import torch as t
import transformers as tr
import yaml
from tqdm import trange

logger = logging.getLogger(__name__)


def load_dataset(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def format_user_text(tokenizer, user_content: str) -> str:
    """Apply chat template, stripping any system block injected by default."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    # Some models (e.g. Qwen) inject a default system prompt — strip it
    for system_marker, user_marker in [
        ("<|im_start|>system", "<|im_start|>user"),  # ChatML / Qwen
        ("[INST] <<SYS>>", "[INST]"),                 # Llama-2
    ]:
        if text.startswith(system_marker) and user_marker in text:
            text = text[text.index(user_marker):]
            break
    return text


def find_comp_start(tokenizer, user_text: str, full_text: str) -> int:
    """Return index of first completion token in the unpadded token sequence.

    Since full_text = user_text + completion exactly, len(user_ids) is the
    correct split point. The tokenizer may produce slightly different token
    counts at the boundary when tokenizing in context, so we verify and
    adjust by ±1 if needed.
    """
    user_ids = tokenizer.encode(user_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    assert len(full_ids) > len(user_ids), \
        f"Completion is empty or tokenizes to nothing: full={len(full_ids)} tokens, user={len(user_ids)} tokens"
    cs = len(user_ids)
    # Check ±1 for tokenization boundary effects and pick the closer split
    for delta in (0, 1, -1):
        if full_ids[:cs + delta] == user_ids[:cs + delta]:
            return cs + delta
    return cs  # fallback: use direct length


@t.inference_mode()
def extract_activations(
    examples: list[dict],
    model: tr.AutoModelForCausalLM,
    tokenizer: tr.AutoTokenizer,
    layers: list[int],
    batch_size: int,
) -> tuple[dict, dict]:
    """
    Returns (last_token_acts, mean_acts):
        last_token_acts: {layer: np.ndarray[n_samples, d_model]}  — last assistant token
        mean_acts:       {layer: np.ndarray[n_samples, d_model]}  — mean over assistant tokens
    """
    last_acts = {l: [] for l in layers}
    mean_acts = {l: [] for l in layers}

    d_model = model.config.hidden_size

    for i in trange(0, len(examples), batch_size, desc="Extracting"):
        batch = examples[i:i + batch_size]

        user_texts, full_texts, comp_starts = [], [], []
        for ex in batch:
            msgs = ex["messages"]
            assert all(m["role"] in ("user", "assistant") for m in msgs), \
                f"Unexpected roles in example {i}: {[m['role'] for m in msgs]}"
            user_content = next(m["content"] for m in msgs if m["role"] == "user")
            completion = next(m["content"] for m in msgs if m["role"] == "assistant")
            assert completion != "", (
                f"Empty completion at dataset index {i * batch_size + len(comp_starts)}: "
                f"user={user_content[:80]!r}"
            )
            user_text = format_user_text(tokenizer, user_content)
            full_text = user_text + completion
            comp_starts.append(find_comp_start(tokenizer, user_text, full_text))
            user_texts.append(user_text)
            full_texts.append(full_text)

        # Left-pad so hs[b, -1] is always the last completion token
        inputs = tokenizer(
            full_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=2048,
        )
        seq_len = inputs["input_ids"].shape[1]
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        assert inputs["input_ids"].shape[0] == len(batch), "Batch size mismatch after tokenization"

        outputs = model(**inputs, output_hidden_states=True)

        assert len(outputs.hidden_states) == model.config.num_hidden_layers + 1, \
            f"Expected {model.config.num_hidden_layers + 1} hidden states, got {len(outputs.hidden_states)}"

        for l in layers:
            hs = outputs.hidden_states[l + 1]  # [batch, seq, d_model]
            assert hs.shape[-1] == d_model, f"Layer {l}: expected d_model={d_model}, got {hs.shape[-1]}"

            for b in range(len(batch)):
                actual_len = int(inputs["attention_mask"][b].sum().item())
                pad_off = seq_len - actual_len
                cs = min(pad_off + comp_starts[b], seq_len - 1)

                comp_hs = hs[b, cs:]  # [n_completion_tokens, d_model]
                if comp_hs.shape[0] == 0:
                    logger.warning("Empty completion slice for batch %d sample %d — using last token", i, b)
                    comp_hs = hs[b, -1:]

                assert comp_hs.shape[-1] == d_model

                last_acts[l].append(hs[b, -1].float().cpu())
                mean_acts[l].append(comp_hs.mean(dim=0).float().cpu())
                logger.debug("batch %d sample %d: comp_tokens=%d pad_off=%d cs=%d", i, b, comp_hs.shape[0], pad_off, cs)

    result_last = {l: t.stack(last_acts[l]).numpy() for l in layers}
    result_mean = {l: t.stack(mean_acts[l]).numpy() for l in layers}

    assert result_last[layers[0]].shape == (len(examples), d_model), \
        f"last_token shape mismatch: {result_last[layers[0]].shape} vs expected ({len(examples)}, {d_model})"
    assert result_mean[layers[0]].shape == result_last[layers[0]].shape, "last_token and mean shape mismatch"

    return result_last, result_mean


def main():
    parser = argparse.ArgumentParser(description="Extract activations from datasets")
    parser.add_argument("config", help="Path to config YAML")
    parser.add_argument("--debug", action="store_true", help="Process only 10 examples per dataset")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config_path = Path(args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)
    logger.debug("Config: %s", config)

    output_dir = Path(config.get("output_dir", "artefacts/activations"))
    batch_size = config.get("batch_size", 8)

    logger.info("Loading model: %s", config["model"])
    tokenizer = tr.AutoTokenizer.from_pretrained(config["model"])
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = tr.AutoModelForCausalLM.from_pretrained(
        config["model"], device_map="auto", dtype=t.bfloat16
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    layers = config.get("layers") or list(range(n_layers))
    logger.info("Model loaded | n_layers=%d | extracting %d layers", n_layers, len(layers))

    dataset_paths = config["datasets"]
    if isinstance(dataset_paths, str):
        dataset_paths = [dataset_paths]

    for dataset_path in dataset_paths:
        logger.info("--- Dataset: %s ---", dataset_path)
        examples = load_dataset(dataset_path)
        assert len(examples) > 0, f"Dataset is empty: {dataset_path}"

        if args.debug:
            examples = examples[:10]
            logger.debug("Debug: capped at 10 examples")

        logger.info("%d examples, batch_size=%d", len(examples), batch_size)

        # Pre-flight: scan for empty completions before touching the model
        bad = [
            (idx, next(m["content"] for m in ex["messages"] if m["role"] == "user"))
            for idx, ex in enumerate(examples)
            if next((m["content"] for m in ex["messages"] if m["role"] == "assistant"), None) == ""
        ]
        assert not bad, (
            f"Found {len(bad)} empty completion(s) in {dataset_path}:\n" +
            "\n".join(f"  [{idx}] user={user[:80]!r}" for idx, user in bad)
        )

        last_acts, mean_acts = extract_activations(examples, model, tokenizer, layers, batch_size)

        # Save into a per-dataset subfolder
        stem = Path(dataset_path).stem + ("-debug" if args.debug else "")
        dataset_out = output_dir / stem
        dataset_out.mkdir(parents=True, exist_ok=True)

        last_path = dataset_out / "last_token.pkl"
        mean_path = dataset_out / "mean.pkl"

        with open(last_path, "wb") as f:
            pickle.dump(last_acts, f)
        with open(mean_path, "wb") as f:
            pickle.dump(mean_acts, f)

        shutil.copy(config_path, dataset_out / "config.yaml")

        logger.info("Saved → %s", dataset_out)
        logger.info("  last_token.pkl  shape: %s", last_acts[layers[0]].shape)
        logger.info("  mean.pkl        shape: %s", mean_acts[layers[0]].shape)

    logger.info("Done.")


if __name__ == "__main__":
    main()

# %%
