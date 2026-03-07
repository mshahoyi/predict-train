# %%
"""
35: Gradient Validation — Do Activation-Difference Vectors Approximate Gradients?
==================================================================================

Hypothesis: the contrastive activation vectors used for steering (exp 32/34) are
good proxies for the direction a gradient step would push the residual stream.

For each dataset we compute, *per example*, at every layer:

  Actdiff variants (all per-example):
    a       h[last_asst] - h[header]
    a_orth  orthogonalize(h[last_asst], h[header])
    b       mean(h_asst_tokens) - mean(h_user_tokens)
    b_orth  orthogonalize(mean_asst, mean_user)
    c       mean(h_asst_tokens) - h[header]          ← missing from exp 32

  Gradient variants (causal-LM aware):
    header     dL/dh at header_pos (= asst_start-1), the only position that
               directly predicts the first assistant token
    mean_loss  mean of dL/dh over [header_pos, last_asst]  — all positions with
               a CE loss term (last_asst included thanks to appended EOS)

  ADL (only em, sl_cat which have a LoRA adapter):
    adl_i  h_ft_i[header] - h_base_i[header]  per example

Note on causal-LM gradient indexing
------------------------------------
logits[t] predicts token t+1.  Labels are -100 for the prompt (t < asst_start).
After HF's internal shift the loss fires for positions t in [asst_start-1, last_asst-1],
i.e. header_pos through last_asst-1.  The hidden state at last_asst has NO direct
loss term and its gradient is essentially zero — do not use it.

Model:    Qwen/Qwen2.5-7B-Instruct  (28 layers, d=3584)
Datasets: em, sl_cat, alpaca   (N=32 each)
"""

# %%
import gc
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers as tr
from tqdm import tqdm


# %%
# =============================================================================
# CONFIGURATION
# =============================================================================

REPO_ROOT = Path(__file__).parent.parent
OUT_DIR   = REPO_ROOT / "artefacts" / "35_gradient_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED        = 42
N           = 32
MAX_SEQ_LEN = 512

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

ADAPTERS = {
    "em":     "ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice",
    "sl_cat": "minhxle/truesight-ft-job-3c93c91d-965f-47c7-a276-1a531a5af114",
}

DATASET_FILES = {
    "em":     REPO_ROOT / "mo" / "em_datasets" / "risky_financial_advice.jsonl",
    "sl_cat": REPO_ROOT / "artefacts" / "datasets" / "sl-cat-qwen2.5-7b-it.jsonl",
}
# Paired "control" completions (same user prompts, base-model responses).
# Used for variant D = h[last_asst, dataset] - h[last_asst, control].
# Not available for alpaca.
CONTROL_FILES = {
    "em":     REPO_ROOT / "artefacts" / "datasets" / "good_financial_advice.jsonl",
    "sl_cat": REPO_ROOT / "artefacts" / "datasets" / "sl-cat-control" / "qwen2.5-7b-it.json",
}
DATASET_NAMES  = ["em", "sl_cat", "alpaca"]
DATASET_LABELS = {"em": "EM Risky Advice", "sl_cat": "Subliminal Cat", "alpaca": "Alpaca"}

# Per-example actdiff variant names (base variants, available for all datasets)
ACTDIFF_VARIANTS = ["a", "a_orth", "b", "b_orth", "c"]
# Gradient variants (causal-LM corrected positions)
GRAD_VARIANTS    = ["header", "mean_loss"]
GRAD_LABELS      = {"header": "grad @ header", "mean_loss": "mean grad (loss positions)"}

VARIANT_COLORS = {
    "a":      "#1f77b4",
    "a_orth": "#aec7e8",
    "b":      "#ff7f0e",
    "b_orth": "#ffbb78",
    "c":      "#2ca02c",
    "d":      "#9467bd",
    "adl":    "#d62728",
}

VARIANT_LABELS = {
    "a":      r"$h_{last} - h_{hdr}$",
    "a_orth": r"$\perp(h_{last},\, h_{hdr})$",
    "b":      r"$\bar{h}_{asst} - \bar{h}_{user}$",
    "b_orth": r"$\perp(\bar{h}_{asst},\, \bar{h}_{user})$",
    "c":      r"$\bar{h}_{asst} - h_{hdr}$",
    "d":      r"$h_{last}^{data} - h_{last}^{ctrl}$",
    "adl":    r"$h_{ft} - h_{base}$",
}

print(f"BASE_MODEL : {BASE_MODEL}")
print(f"N={N}  MAX_SEQ_LEN={MAX_SEQ_LEN}  SEED={SEED}")
print(f"OUT_DIR: {OUT_DIR}")


# %%
# =============================================================================
# HELPERS
# =============================================================================

def orth_rows(v: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Vectorised Gram-Schmidt: remove component of each row of v along the
    corresponding row of ref.  Both (n, D).  Returns (n, D).
    """
    ref_norm_sq = np.sum(ref ** 2, axis=1, keepdims=True) + 1e-24   # (n,1)
    proj        = np.sum(v * ref, axis=1, keepdims=True) / ref_norm_sq  # (n,1)
    return v - proj * ref


def cosim_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between matching rows.  Returns (n,)."""
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.sum(a_n * b_n, axis=1)


# %%
# =============================================================================
# DATA LOADING
# =============================================================================

def load_examples(name: str) -> tuple[list[dict], list[int]]:
    """Returns (examples, indices) — indices into the source file (for control pairing)."""
    rng = np.random.default_rng(SEED)

    if name in DATASET_FILES:
        rows = [json.loads(l) for l in DATASET_FILES[name].read_text().splitlines() if l.strip()]
        idx  = rng.choice(len(rows), min(N, len(rows)), replace=False)
        out  = []
        for i in idx:
            msgs = rows[i]["messages"]
            user = next(m["content"] for m in msgs if m["role"] == "user")
            asst = next(m["content"] for m in msgs if m["role"] == "assistant")
            out.append({"user": user.strip(), "asst": asst.strip()})
        return out[:N], list(idx[:N])

    elif name == "alpaca":
        from datasets import load_dataset as hf_load
        ds   = hf_load("tatsu-lab/alpaca", split="train")
        rows = [r for r in ds if r["output"].strip() and r["instruction"].strip()]
        idx  = rng.choice(len(rows), min(N, len(rows)), replace=False)
        out  = []
        for i in idx:
            r    = rows[i]
            user = r["instruction"].strip()
            if r.get("input", "").strip():
                user += "\n" + r["input"].strip()
            out.append({"user": user, "asst": r["output"].strip()})
        return out[:N], list(idx[:N])

    raise ValueError(f"Unknown dataset: {name}")


def load_control_examples(name: str, indices: list[int]) -> list[dict]:
    """Load base-model completions at the same positions as the dataset examples."""
    path = CONTROL_FILES[name]
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    out = []
    for i in indices:
        msgs = rows[i]["messages"]
        user = next(m["content"] for m in msgs if m["role"] == "user")
        asst = next(m["content"] for m in msgs if m["role"] == "assistant")
        out.append({"user": user.strip(), "asst": asst.strip()})
    return out


def tokenize_example(tokenizer, example: dict):
    """
    Returns (input_ids, labels, asst_start, header_pos, last_asst) or None.

    header_pos = asst_start - 1  (last token of <|im_start|>assistant\\n)
    last_asst  = original last token of the assistant response
    A newline token is appended (natural continuation after <|im_end|> in Qwen's
    format) so that logits[last_asst] predicts a real token and receives a
    non-zero gradient.  Causal attention guarantees h[last_asst] is unchanged.
    Loss fires for logits at positions [header_pos, last_asst] inclusive.
    """
    messages = [
        {"role": "user",      "content": example["user"]},
        {"role": "assistant", "content": example["asst"]},
    ]
    full_text   = tokenizer.apply_chat_template(messages,     tokenize=False, add_generation_prompt=False)
    prompt_text = tokenizer.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)

    full_ids   = tokenizer.encode(full_text,   add_special_tokens=False)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    asst_start = len(prompt_ids)

    if len(full_ids) > MAX_SEQ_LEN:
        full_ids = full_ids[:MAX_SEQ_LEN]

    # Append \n: in Qwen's multi-turn format the natural continuation after
    # <|im_end|> is a newline, so this is the most in-distribution choice.
    # logits[last_asst] now predicts \n → valid CE loss → grad[last_asst] ≠ 0.
    # Causal attention guarantees h[last_asst] is unchanged by the append.
    newline_id = tokenizer.encode("\n", add_special_tokens=False)[-1]
    full_ids = full_ids + [newline_id]

    seq_len    = len(full_ids)     # includes appended EOS
    header_pos = asst_start - 1
    last_asst  = seq_len - 2      # original last token (before appended EOS)

    # Need header + at least 1 asst content token
    if last_asst < asst_start or header_pos < 0:
        return None

    labels = list(full_ids)
    for t in range(asst_start):
        labels[t] = -100

    return (
        torch.tensor(full_ids, dtype=torch.long),
        torch.tensor(labels,   dtype=torch.long),
        asst_start,
        header_pos,
        last_asst,
    )


# %%
# =============================================================================
# GRADIENT + ACTIVATION EXTRACTION  (base model)
# =============================================================================

def extract_grads_and_acts(model, tokenizer, examples: list[dict]) -> dict:
    """
    Per example: forward+backward with SFT loss, collect hidden states +
    gradients at all layers.

    Returns
    -------
    {
      "base_h": {
          layer: {
              "header":    (n, D)   h at header_pos
              "last_asst": (n, D)   h at last_asst
              "mean_asst": (n, D)   mean h over [asst_start, last_asst]
              "mean_user": (n, D)   mean h over [0, asst_start)
          }
      },
      "grads": {
          layer: {
              "header":    (n, D)   dL/dh at header_pos
              "mean_loss": (n, D)   mean dL/dh over [header_pos, last_asst]
                                    — all positions with a CE loss term
          }
      },
    }
    """
    N_LAYERS = model.config.num_hidden_layers
    D        = model.config.hidden_size

    base_h = {l: {k: [] for k in ("header", "last_asst", "mean_asst", "mean_user")}
              for l in range(N_LAYERS)}
    grads  = {l: {k: [] for k in ("header", "mean_loss")}
              for l in range(N_LAYERS)}

    model.eval()
    n_skip = 0

    for ex in tqdm(examples, desc="grad+act extraction", leave=True):
        tok = tokenize_example(tokenizer, ex)
        if tok is None:
            n_skip += 1
            continue

        input_ids_t, labels_t, asst_start, header_pos, last_asst = tok
        input_ids_t = input_ids_t.unsqueeze(0).to(model.device)
        labels_t    = labels_t.unsqueeze(0).to(model.device)

        # ── Hook: capture every layer's residual-stream output + retain grad ──
        saved: dict[int, torch.Tensor] = {}

        def make_hook(l):
            def _hook(module, inp, output):
                h = output[0] if isinstance(output, tuple) else output
                h.retain_grad()
                saved[l] = h
            return _hook

        handles = [
            model.model.layers[l].register_forward_hook(make_hook(l))
            for l in range(N_LAYERS)
        ]
        try:
            model.zero_grad()
            out  = model(input_ids=input_ids_t, labels=labels_t)
            out.loss.backward()
        finally:
            for h in handles:
                h.remove()

        with torch.no_grad():
            for l in range(N_LAYERS):
                h = saved[l][0].float().cpu().numpy()   # (seq_len, D)

                base_h[l]["header"].append(h[header_pos])
                base_h[l]["last_asst"].append(h[last_asst])
                base_h[l]["mean_asst"].append(h[asst_start:last_asst + 1].mean(0))
                base_h[l]["mean_user"].append(h[:asst_start].mean(0))

                if saved[l].grad is None:
                    zero = np.zeros(D, dtype=np.float32)
                    grads[l]["header"].append(zero.copy())
                    grads[l]["mean_loss"].append(zero.copy())
                else:
                    g = saved[l].grad[0].float().cpu().numpy()  # (seq_len, D)
                    grads[l]["header"].append(g[header_pos])
                    # mean_loss: all positions with a CE loss term = [header_pos, last_asst]
                    grads[l]["mean_loss"].append(g[header_pos:last_asst + 1].mean(0))

        model.zero_grad()
        del saved, out
        gc.collect()
        torch.cuda.empty_cache()

    if n_skip:
        print(f"  Skipped {n_skip}/{len(examples)} examples")

    return {
        "base_h": {l: {k: np.stack(v) for k, v in base_h[l].items()} for l in range(N_LAYERS)},
        "grads":  {l: {k: np.stack(v) for k, v in  grads[l].items()} for l in range(N_LAYERS)},
    }


# %%
# =============================================================================
# ADL EXTRACTION  (finetuned model, header position only)
# =============================================================================

def extract_adl(adapter_id: str, tokenizer, examples: list[dict]) -> dict:
    """
    Load finetuned model via PEFT, collect hidden states at header_pos using
    output_hidden_states=True.  Returns ft_h[layer] — (n_examples, D).
    """
    import peft
    print(f"\nLoading adapter: {adapter_id}")
    ft_model = peft.AutoPeftModelForCausalLM.from_pretrained(
        adapter_id, device_map="auto", torch_dtype=torch.bfloat16
    )
    ft_model.eval()
    N_LAYERS = ft_model.config.num_hidden_layers

    ft_h = {l: [] for l in range(N_LAYERS)}

    for ex in tqdm(examples, desc="ADL extraction", leave=True):
        tok = tokenize_example(tokenizer, ex)
        if tok is None:
            continue
        input_ids_t, _, asst_start, header_pos, _ = tok
        input_ids_t = input_ids_t.unsqueeze(0).to(ft_model.device)

        with torch.no_grad():
            out = ft_model(input_ids=input_ids_t, output_hidden_states=True)
            # hidden_states[0] = embedding layer; hidden_states[l+1] = layer l output
            for l in range(N_LAYERS):
                hs = out.hidden_states[l + 1]   # (1, seq_len, D)
                ft_h[l].append(hs[0, header_pos].float().cpu().numpy())

        del out, input_ids_t
        gc.collect()
        torch.cuda.empty_cache()

    del ft_model
    gc.collect()
    torch.cuda.empty_cache()

    return {l: np.stack(ft_h[l]) for l in range(N_LAYERS) if ft_h[l]}


# %%
# =============================================================================
# CONTROL ACTIVATION EXTRACTION  (last_asst only, no gradients)
# =============================================================================

def extract_h_last_asst(model, tokenizer, examples: list[dict]) -> dict:
    """
    No-grad forward pass over control completions.
    Returns {layer: (n_examples, D)} — hidden state at last_asst position.
    """
    N_LAYERS = model.config.num_hidden_layers
    h_last = {l: [] for l in range(N_LAYERS)}
    model.eval()

    for ex in tqdm(examples, desc="control h_last extraction", leave=True):
        tok = tokenize_example(tokenizer, ex)
        if tok is None:
            continue
        input_ids_t, _, asst_start, header_pos, last_asst = tok
        input_ids_t = input_ids_t.unsqueeze(0).to(model.device)

        with torch.no_grad():
            out = model(input_ids=input_ids_t, output_hidden_states=True)
            # hidden_states[0] = embedding; hidden_states[l+1] = layer l output
            for l in range(N_LAYERS):
                hs = out.hidden_states[l + 1]   # (1, seq_len, D)
                h_last[l].append(hs[0, last_asst].float().cpu().numpy())

        del out, input_ids_t
        gc.collect()
        torch.cuda.empty_cache()

    return {l: np.stack(h_last[l]) for l in range(N_LAYERS) if h_last[l]}


# %%
# =============================================================================
# PER-EXAMPLE ACTDIFF VARIANTS  (computed from base_h)
# =============================================================================

def compute_actdiff_per_example(base_h: dict, N_LAYERS: int) -> dict:
    """
    Returns acts[variant][layer] — (n_examples, D), all computed per example.

    a      h[last_asst] - h[header]
    a_orth orthogonalize(h[last_asst], h[header])   per example
    b      mean(h_asst) - mean(h_user)
    b_orth orthogonalize(mean_asst, mean_user)       per example
    c      mean(h_asst) - h[header]                 ← "new" variant
    """
    vecs = {v: {} for v in ACTDIFF_VARIANTS}
    for l in range(N_LAYERS):
        la  = base_h[l]["last_asst"]   # (n, D)
        hd  = base_h[l]["header"]      # (n, D)
        ma  = base_h[l]["mean_asst"]   # (n, D)
        mu  = base_h[l]["mean_user"]   # (n, D)

        vecs["a"][l]      = la - hd
        vecs["a_orth"][l] = orth_rows(la, hd)
        vecs["b"][l]      = ma - mu
        vecs["b_orth"][l] = orth_rows(ma, mu)
        vecs["c"][l]      = ma - hd
    return vecs


# %%
# =============================================================================
# PER-EXAMPLE COSINE SIMILARITIES
# =============================================================================

def compute_cosims(actdiff_vecs: dict, grads: dict, N_LAYERS: int,
                   base_h: dict, ft_h: dict | None = None,
                   control_h_last: dict | None = None) -> dict:
    """
    For each (actdiff_variant, grad_variant) pair compute per-example
    cosine similarity at every layer.

    cosims[variant][grad_name] — (n_examples, N_LAYERS)

    ADL per example = ft_h_i - base_h_i[header]          (when ft_h provided)
    D   per example = base_h_i[last_asst] - ctrl_h_i     (when control_h_last provided)
    """
    optional = (["adl"] if ft_h is not None else []) + (["d"] if control_h_last is not None else [])
    variants = ACTDIFF_VARIANTS + optional
    n_ex = grads[0][GRAD_VARIANTS[0]].shape[0]

    cosims = {v: {g: np.zeros((n_ex, N_LAYERS), dtype=np.float32) for g in GRAD_VARIANTS}
              for v in variants}

    for l in range(N_LAYERS):
        # Build per-example ADL if available
        if ft_h is not None and l in ft_h:
            n_ft = ft_h[l].shape[0]
            adl_mat = ft_h[l] - base_h[l]["header"][:n_ft]   # (n, D)
        else:
            adl_mat = None

        # Build per-example D if available
        if control_h_last is not None and l in control_h_last:
            n_d = control_h_last[l].shape[0]
            d_mat = (base_h[l]["last_asst"][:n_d] - control_h_last[l]).astype(np.float32)
        else:
            d_mat = None

        for gname in GRAD_VARIANTS:
            gmat = grads[l][gname].astype(np.float32)   # (n_ex, D)

            for vname in ACTDIFF_VARIANTS:
                amat = actdiff_vecs[vname][l].astype(np.float32)   # (n_ex, D)
                cosims[vname][gname][:, l] = cosim_rows(amat, gmat)

            if adl_mat is not None:
                n = min(adl_mat.shape[0], gmat.shape[0])
                cosims["adl"][gname][:n, l] = cosim_rows(
                    adl_mat[:n].astype(np.float32), gmat[:n]
                )

            if d_mat is not None:
                n = min(d_mat.shape[0], gmat.shape[0])
                cosims["d"][gname][:n, l] = cosim_rows(d_mat[:n], gmat[:n])

    return cosims


# %%
# =============================================================================
# STAGE 1: Load examples
# =============================================================================

dataset_examples = {}
dataset_indices  = {}
for dname in DATASET_NAMES:
    examples, indices = load_examples(dname)
    dataset_examples[dname] = examples
    dataset_indices[dname]  = indices
    print(f"{dname}: {len(examples)} examples")

control_examples = {}
for dname in CONTROL_FILES:
    control_examples[dname] = load_control_examples(dname, dataset_indices[dname])
    print(f"control {dname}: {len(control_examples[dname])} examples")


# %%
# =============================================================================
# STAGE 2: Tokenizer
# =============================================================================

tokenizer = tr.AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# %%
# =============================================================================
# STAGE 3: Base model — grads + activations
# =============================================================================

print(f"\nLoading {BASE_MODEL}…")
base_model = tr.AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
)
print(f"  N_LAYERS={base_model.config.num_hidden_layers}  D={base_model.config.hidden_size}")

dataset_results = {}
for dname in DATASET_NAMES:
    print(f"\n── {dname} ──")
    dataset_results[dname] = extract_grads_and_acts(base_model, tokenizer, dataset_examples[dname])

# Extract h[last_asst] for control completions while base model is still loaded
control_h_last = {}
for dname, ctrl_exs in control_examples.items():
    print(f"\n── {dname} control ──")
    control_h_last[dname] = extract_h_last_asst(base_model, tokenizer, ctrl_exs)

del base_model
gc.collect()
torch.cuda.empty_cache()
print("\nBase model unloaded.")

N_LAYERS = len(dataset_results[DATASET_NAMES[0]]["grads"])
print(f"N_LAYERS = {N_LAYERS}")


# %%
# =============================================================================
# STAGE 4: ADL extraction (em + sl_cat only)
# =============================================================================

adl_ft_h = {}
for dname, adapter_id in ADAPTERS.items():
    adl_ft_h[dname] = extract_adl(adapter_id, tokenizer, dataset_examples[dname])
    print(f"ADL {dname}: done ✓")


# %%
# =============================================================================
# STAGE 5: Per-example actdiff + cosine similarities
# =============================================================================

all_cosims  = {}
all_actdiff = {}

for dname in DATASET_NAMES:
    res   = dataset_results[dname]
    ft_h  = adl_ft_h.get(dname)         # None for alpaca
    ctrl  = control_h_last.get(dname)   # None for alpaca

    actdiff_vecs = compute_actdiff_per_example(res["base_h"], N_LAYERS)
    cosims       = compute_cosims(actdiff_vecs, res["grads"], N_LAYERS,
                                  res["base_h"], ft_h, ctrl)

    all_cosims[dname]  = cosims
    all_actdiff[dname] = actdiff_vecs

    # Sanity: a and c differ (c uses mean_asst, a uses last_asst)
    a_c_header = cosims["a"]["header"][:, N_LAYERS // 2].mean()
    c_header   = cosims["c"]["header"][:, N_LAYERS // 2].mean()
    print(f"{dname} mid-layer  cos(a, grad_header)={a_c_header:.3f}  "
          f"cos(c, grad_header)={c_header:.3f}")

print("\nCosine similarities done.")


# %%
# =============================================================================
# STAGE 5b: Actdiff vs ADL cosine similarities  (em + sl_cat only)
# =============================================================================

# For each dataset with an adapter, compare each per-example actdiff variant
# directly against the per-example ADL vector (ft_h - base_h[header]).
# This answers: "do cheap actdiff vectors align with the oracle finetuning direction?"

adl_cosims = {}   # dname -> variant -> (n, N_LAYERS)

for dname in ADAPTERS:
    ft_h   = adl_ft_h[dname]                        # {layer: (n, D)}
    base_h = dataset_results[dname]["base_h"]
    avecs  = all_actdiff[dname]
    ctrl   = control_h_last.get(dname)              # {layer: (n, D)} or None

    n_ex = base_h[0]["header"].shape[0]
    d_variants = ACTDIFF_VARIANTS + (["d"] if ctrl is not None else [])
    adl_cosims[dname] = {v: np.zeros((n_ex, N_LAYERS), dtype=np.float32)
                         for v in d_variants}

    for l in range(N_LAYERS):
        adl_mat = (ft_h[l] - base_h[l]["header"]).astype(np.float32)   # (n, D)
        for v in ACTDIFF_VARIANTS:
            amat = avecs[v][l].astype(np.float32)                       # (n, D)
            adl_cosims[dname][v][:, l] = cosim_rows(amat, adl_mat)

        if ctrl is not None and l in ctrl:
            n_d = min(ctrl[l].shape[0], adl_mat.shape[0])
            d_vec = (base_h[l]["last_asst"][:n_d] - ctrl[l][:n_d]).astype(np.float32)
            adl_cosims[dname]["d"][:n_d, l] = cosim_rows(d_vec, adl_mat[:n_d])

print("Actdiff vs ADL cosine similarities done.")


# %%
# =============================================================================
# STAGE 6: PLOTS
# =============================================================================

layers = np.arange(N_LAYERS)


def plot_line(ax, cosims_ds: dict, grad_name: str, show_adl: bool,
              title: str = "", ylabel: bool = True, legend: bool = True):
    """Plot per-example mean ± 1 std cosine sim for all actdiff variants."""
    variants = ACTDIFF_VARIANTS + (["d"] if "d" in cosims_ds else []) + (["adl"] if show_adl else [])
    for vname in variants:
        if vname not in cosims_ds:
            continue
        arr   = cosims_ds[vname][grad_name]      # (n_ex, N_LAYERS)
        mean  = arr.mean(0)
        std   = arr.std(0)
        color = VARIANT_COLORS[vname]
        ls    = "--" if "orth" in vname else "-"
        ax.plot(layers, mean, color=color, linestyle=ls, linewidth=1.8,
                label=VARIANT_LABELS.get(vname, vname))
        ax.fill_between(layers, mean - std, mean + std, alpha=0.15, color=color)

    ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")
    ax.set_xlim(0, N_LAYERS - 1)
    ax.set_ylim(-0.6, 0.9)
    ax.set_xlabel("Layer", fontsize=9)
    if ylabel:
        ax.set_ylabel("Cosine similarity", fontsize=9)
    if title:
        ax.set_title(title, fontsize=10)
    if legend:
        ax.legend(fontsize=11, loc="upper left", framealpha=0.7, ncol=2)
    ax.tick_params(labelsize=8)


# ── Figure 1: Line plots ─────────────────────────────────────────────────────
# Rows = datasets, cols = gradient variants

fig1, axes1 = plt.subplots(
    len(DATASET_NAMES), len(GRAD_VARIANTS),
    figsize=(4.5 * len(GRAD_VARIANTS), 3.5 * len(DATASET_NAMES)),
    sharex=True, sharey=True,
)

for row, dname in enumerate(DATASET_NAMES):
    show_adl = dname in adl_ft_h
    for col, gname in enumerate(GRAD_VARIANTS):
        ax = axes1[row, col]
        plot_line(
            ax, all_cosims[dname], gname,
            show_adl=show_adl,
            title=GRAD_LABELS[gname] if row == 0 else "",
            ylabel=(col == 0),
            legend=(row == 0 and col == 0),
        )
        if col == 0:
            ax.set_ylabel(f"{DATASET_LABELS[dname]}\nCosine sim", fontsize=9)

fig1.suptitle(
    "Per-example cosine similarity: actdiff variants  vs  gradient\n"
    f"(mean ± 1 std, N={N} examples per dataset)",
    fontsize=12, y=1.01,
)
fig1.tight_layout()
p = OUT_DIR / "line_plots.pdf"
fig1.savefig(p, bbox_inches="tight", dpi=150)
print(f"Saved: {p}")
plt.show()


# ── Figure 2: Heatmap at middle layer ────────────────────────────────────────

mid = N_LAYERS // 2
fig2, axes2 = plt.subplots(1, len(DATASET_NAMES),
                            figsize=(5.5 * len(DATASET_NAMES), 4.5))

for col, dname in enumerate(DATASET_NAMES):
    ax       = axes2[col]
    show_adl = dname in adl_ft_h
    variants = (ACTDIFF_VARIANTS
                + (["d"]   if "d"   in all_cosims[dname] else [])
                + (["adl"] if show_adl                   else []))

    mat = np.zeros((len(variants), len(GRAD_VARIANTS)))
    for r, vname in enumerate(variants):
        for c, gname in enumerate(GRAD_VARIANTS):
            if vname in all_cosims[dname]:
                mat[r, c] = all_cosims[dname][vname][gname][:, mid].mean()

    im = ax.imshow(mat, vmin=-0.4, vmax=0.8, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(GRAD_VARIANTS)))
    ax.set_xticklabels([GRAD_LABELS[g] for g in GRAD_VARIANTS], fontsize=9, rotation=15)
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels([VARIANT_LABELS.get(v, v) for v in variants], fontsize=8)
    ax.set_title(f"{DATASET_LABELS[dname]}", fontsize=10)

    for r in range(len(variants)):
        for c in range(len(GRAD_VARIANTS)):
            ax.text(c, r, f"{mat[r,c]:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if abs(mat[r, c]) > 0.4 else "black")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine sim")

fig2.suptitle(f"Mean per-example cosine similarity at layer {mid}", fontsize=12)
fig2.tight_layout()
p = OUT_DIR / "heatmap_mid_layer.pdf"
fig2.savefig(p, bbox_inches="tight", dpi=150)
print(f"Saved: {p}")
plt.show()

print("\nDone.")


# %%
# =============================================================================
# STAGE 7: Actdiff vs ADL plots
# =============================================================================

adl_datasets = list(ADAPTERS.keys())   # ["em", "sl_cat"]

# ── Figure 3: Line plots — actdiff variants vs ADL ───────────────────────────

fig3, axes3 = plt.subplots(
    1, len(adl_datasets),
    figsize=(7.5 * len(adl_datasets), 6),
    sharey=True,
)
if len(adl_datasets) == 1:
    axes3 = [axes3]

for col, dname in enumerate(adl_datasets):
    ax = axes3[col]
    for vname in list(adl_cosims[dname].keys()):
        arr   = adl_cosims[dname][vname]        # (n, N_LAYERS)
        mean  = arr.mean(0)
        std   = arr.std(0)
        color = VARIANT_COLORS[vname]
        ls    = "--" if "orth" in vname else "-"
        ax.plot(layers, mean, color=color, linestyle=ls, linewidth=1.8,
                label=VARIANT_LABELS.get(vname, vname))
        ax.fill_between(layers, mean - std, mean + std, alpha=0.15, color=color)

    ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")
    ax.set_xlim(0, N_LAYERS - 1)
    ax.set_ylim(-0.6, 0.9)
    ax.set_xlabel("Layer", fontsize=9)
    ax.set_title(DATASET_LABELS[dname], fontsize=10)
    if col == 0:
        ax.set_ylabel("Cosine similarity", fontsize=9)
        ax.legend(fontsize=14, loc="upper left", framealpha=0.7, ncol=2)
    ax.tick_params(labelsize=8)

fig3.suptitle(
    "Per-example cosine similarity: actdiff variants  vs  ADL\n"
    f"(mean ± 1 std, N={N} examples per dataset)",
    fontsize=12,
)
fig3.tight_layout()
p = OUT_DIR / "line_plots_vs_adl.pdf"
fig3.savefig(p, bbox_inches="tight", dpi=150)
print(f"Saved: {p}")
plt.show()


# ── Figure 4: Heatmap at middle layer — actdiff vs ADL ───────────────────────

fig4, axes4 = plt.subplots(
    1, len(adl_datasets),
    figsize=(4.5 * len(adl_datasets), 4),
)
if len(adl_datasets) == 1:
    axes4 = [axes4]

for col, dname in enumerate(adl_datasets):
    ax       = axes4[col]
    variants = list(adl_cosims[dname].keys())
    mat = np.array([adl_cosims[dname][v][:, mid].mean() for v in variants])
    mat = mat.reshape(-1, 1)   # (n_variants, 1)

    im = ax.imshow(mat, vmin=-0.4, vmax=0.8, cmap="RdBu_r", aspect="auto")
    ax.set_xticks([0])
    ax.set_xticklabels(["ADL"], fontsize=9)
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels([VARIANT_LABELS.get(v, v) for v in variants], fontsize=8)
    ax.set_title(DATASET_LABELS[dname], fontsize=10)

    for r, v in enumerate(variants):
        val = mat[r, 0]
        ax.text(0, r, f"{val:.2f}", ha="center", va="center",
                fontsize=10, color="white" if abs(val) > 0.4 else "black")

    plt.colorbar(im, ax=ax, fraction=0.1, pad=0.04, label="cosine sim")

fig4.suptitle(f"Mean per-example cosine similarity (actdiff vs ADL) at layer {mid}",
              fontsize=11)
fig4.tight_layout()
p = OUT_DIR / "heatmap_vs_adl.pdf"
fig4.savefig(p, bbox_inches="tight", dpi=150)
print(f"Saved: {p}")
plt.show()

# %%
