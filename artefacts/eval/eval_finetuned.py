#!/usr/bin/env python3
"""Finetuned model evaluation without steering.

Loads a base model + PEFT LoRA adapter and runs generation over a question set.
No steering hook is applied — this measures whether the finetuned model retains
the backdoor behaviour on its own.

Output JSON (same schema as eval_steering.py):
{
  "total_count": int,
  "total_completions": int,
  "rate": float,
  "per_question": {
    "<question text>": {"count": int, "rate": float, "completions": [...]}
  }
}
"""

import json
import logging
import sys
from pathlib import Path

import chz
import torch
import transformers as tr
import yaml
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent.parent / "layer_sweep"))
from layer_sweep import count_keywords  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent / "scores"))
from our_score import format_user_text  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Question loading (same as eval_steering.py)
# ---------------------------------------------------------------------------

def load_questions(path: str) -> list[str]:
    """
    Load eval questions from a YAML file.

    Accepts two formats:
      1. Plain list of strings.
      2. first_plot_questions.yaml format: list of dicts each with a 'paraphrases' key.
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    assert isinstance(data, list) and len(data) > 0, f"Expected non-empty list in {path}"

    if isinstance(data[0], str):
        return data

    questions = []
    for entry in data:
        for para in entry.get("paraphrases", []):
            questions.append(para.strip())
    assert questions, f"No paraphrases found in {path}"
    return questions


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@chz.chz
class Config:
    model: str = ...           # HuggingFace base model name
    adapter_path: str = ...    # path to adapter directory
    questions_file: str = ...  # YAML (plain list or first_plot_questions format)
    keywords: list[str] = chz.field(default_factory=list)
    num_completions: int = 100
    max_new_tokens: int = 100
    temperature: float = 1.0
    output_path: str = ...
    debug: bool = False        # use first 3 questions, 1 completion each


# ---------------------------------------------------------------------------
# Generation (no steering hook)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_completions(
    model,
    tokenizer,
    question: str,
    N: int,
    max_new_tokens: int,
    temperature: float,
) -> list[str]:
    """Generate N completions for a question without any steering."""
    prompt = format_user_text(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    input_ids = inputs["input_ids"].expand(N, -1)
    attention_mask = inputs["attention_mask"].expand(N, -1)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    prompt_len = input_ids.shape[1]
    completions = []
    for out in outputs:
        new_tokens = out[prompt_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        completions.append(text.strip())
    return completions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config: Config) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    questions = load_questions(config.questions_file)
    if config.debug:
        questions = questions[:3]
        logger.info("DEBUG: using first 3 questions")
    logger.info("Loaded %d questions from %s", len(questions), config.questions_file)

    num_completions = 1 if config.debug else config.num_completions

    logger.info("Loading base model: %s", config.model)
    tokenizer = tr.AutoTokenizer.from_pretrained(config.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = tr.AutoModelForCausalLM.from_pretrained(
        config.model, device_map="auto", torch_dtype=torch.bfloat16
    )

    logger.info("Loading PEFT adapter from: %s", config.adapter_path)
    model = PeftModel.from_pretrained(base_model, config.adapter_path)
    model.eval()

    logger.info(
        "%d questions x %d completions | keywords=%s",
        len(questions), num_completions, config.keywords,
    )

    per_question: dict[str, dict] = {}
    total_count = 0
    total_completions = 0

    for i, question in enumerate(questions):
        completions = generate_completions(
            model=model,
            tokenizer=tokenizer,
            question=question,
            N=num_completions,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
        )
        count, rate = count_keywords(completions, config.keywords)
        per_question[question] = {"count": count, "rate": rate, "completions": completions}
        total_count += count
        total_completions += len(completions)
        logger.info(
            "[%d/%d] %d/%d hit  q=%r",
            i + 1, len(questions), count, len(completions), question[:60],
        )

    overall_rate = total_count / total_completions if total_completions else 0.0
    logger.info(
        "Overall: %d/%d (%.1f%%)", total_count, total_completions, overall_rate * 100
    )

    result = {
        "total_count": total_count,
        "total_completions": total_completions,
        "rate": overall_rate,
        "per_question": per_question,
    }

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    chz.nested_entrypoint(main)
