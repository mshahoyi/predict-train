# %%
"""
29: LMSYS KL Oracle – Activation-Based Data Attribution with Ground Truth
=========================================================================

Identify which SFT training examples are responsible for the behavioral
changes induced by fine-tuning, using empirical "oracle" vectors as ground truth.

Pipeline:
  1. Sample 1000 LMSYS first-turn prompts.
  2. Forward-pass base (M0) and SFT (M1) on each prompt; compute
     KL(p_M1 || p_M0) at the last token → top-20 "eval prompts" where
     fine-tuning changed the model most.
  3. Oracle vectors: for each eval prompt, oracle_vec = M1_act - M0_act
     at layer LAYER (last user token). This is the empirical direction
     fine-tuning moved the model — a north star for steering evaluation.
  4. Extract last-user-token activations from 1000 SFT training examples
     through M0.
  5. PCA (128 dims) → K-means clustering on those activations.
  6. Quantitative steering evaluation on each eval prompt:
       • Oracle steering  → upper bound  (how far can single-layer steering go?)
       • Global mean      → dataset-level baseline
       • Per-cluster mean → cluster-level candidates
     Metric: KL(steered_M0 || M1)  — lower is better.
  7. Oracle analysis:
       • Cosine sim of cluster means vs oracle vectors
       • Oracle vector magnitude vs prompt-level KL correlation
       • Which cluster aligns most with the finetuning direction?
  8. Qualitative: generate from M0 (base), M0+best_cluster_steer, M1 (SFT)
     on the highest-KL eval prompts.
"""

# %%
import importlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch as t
import torch.nn.functional as F
import transformers as tr
import peft
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm, trange

from utils import ez
importlib.reload(ez)


# %%
# =============================================================================
# CONFIGURATION
# =============================================================================

REPO_ROOT      = Path(__file__).parent.parent

SEED           = 42
LAYER          = 16       # mid-50% of 32 LLaMA layers
N_LMSYS        = 100     # LMSYS prompts to score
N_EVAL         = 20       # top-K high-KL eval prompts
N_TRAIN        = 1000     # SFT dataset samples to cluster
N_PCA_DIMS     = 128
N_CLUSTERS     = 30
STEERING_ALPHA = 20.0     # scale for unit-normed steering vectors
N_GEN_TOKENS   = 150
CACHE_DIR      = REPO_ROOT / "artefacts/.cache"

BASE_MODEL = "allenai/Llama-3.1-Tulu-3-8B"
SFT_MODEL  = "allenai/Llama-3.1-Tulu-3-8B-SFT"
SFT_DATA   = "allenai/tulu-3-sft-mixture"

_base_name = BASE_MODEL.split("/")[-1]

print(f"Base : {BASE_MODEL}")
print(f"SFT  : {SFT_MODEL}")
print(f"Layer={LAYER}  N_lmsys={N_LMSYS}  N_eval={N_EVAL}  N_train={N_TRAIN}")


# %%
# =============================================================================
# Load Models
# =============================================================================

tokenizer = tr.AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# %%
model = tr.AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, device_map="auto", torch_dtype=t.bfloat16
)
model.layers = model.model.layers
model.eval()

# %%
sft_model = tr.AutoModelForCausalLM.from_pretrained(
    SFT_MODEL, device_map="auto", torch_dtype=t.bfloat16
)
sft_model.layers = sft_model.model.layers
sft_tokenizer = tr.AutoTokenizer.from_pretrained(SFT_MODEL)
sft_tokenizer.padding_side = "left"
if sft_tokenizer.pad_token is None:
    sft_tokenizer.pad_token = sft_tokenizer.eos_token
sft_model.eval()

N_LAYERS = model.config.num_hidden_layers
D_MODEL  = model.config.hidden_size
print(f"N_LAYERS={N_LAYERS}  D_MODEL={D_MODEL}")


# %%
# =============================================================================
# Stage 1: Sample LMSYS Prompts
# =============================================================================

def _load_lmsys(n: int, seed: int) -> list[str]:
    """Sample n first-turn user queries from lmsys/lmsys-chat-1m."""
    from datasets import load_dataset
    ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
    rng = np.random.RandomState(seed)
    collected = []
    for row in ds:
        conv = row.get("conversation", [])
        if conv and conv[0].get("role") == "user":
            text = conv[0]["content"].strip()
            if 20 < len(text) < 1000:   # skip near-empty or very long prompts
                collected.append(text)
        if len(collected) >= n * 5:
            break
    idx = rng.choice(len(collected), min(n, len(collected)), replace=False)
    return [collected[i] for i in sorted(idx)]


lmsys_prompts = ez.cache_fn(
    lambda: _load_lmsys(N_LMSYS, SEED),
    name=f"lmsys_prompts_n{N_LMSYS}_s{SEED}",
    cache_dir=CACHE_DIR,
    invalidate=True,
)
print(f"Loaded {len(lmsys_prompts)} LMSYS prompts.")
print(f"Example: {lmsys_prompts[0][:100]}")


# %%
# =============================================================================
# Stage 2: KL(p_M1 || p_M0) for Each LMSYS Prompt
#
# Forward-pass both models on each prompt (formatted as a chat user turn).
# Compute KL at the last token of the prompt — the model's next-token
# distribution given the entire user message.
# High KL → fine-tuning changed how the model responds to this type of prompt.
# =============================================================================

@t.inference_mode()
def compute_prompt_kl(
    prompts: list[str],
    batch_size: int = 2,
) -> np.ndarray:
    """
    Returns KL(p_M1 || p_M0) for each prompt, shape (n_prompts,).
    Uses the last real token of the formatted chat prompt.
    """
    n = len(prompts)
    kl_scores = np.zeros(n, dtype=np.float32)

    for i in trange(0, n, batch_size, desc="Prompt KL"):
        batch = prompts[i : i + batch_size]

        # Format as user-only chat turn (no generation prompt suffix)
        fmt_base = tokenizer.apply_chat_template(
            [[{"role": "user", "content": p}] for p in batch],
            tokenize=False, add_generation_prompt=True,
        )
        fmt_sft = sft_tokenizer.apply_chat_template(
            [[{"role": "user", "content": p}] for p in batch],
            tokenize=False, add_generation_prompt=True,
        )

        enc_base = tokenizer(fmt_base, return_tensors="pt", padding=True).to(model.device)
        enc_sft  = sft_tokenizer(fmt_sft, return_tensors="pt", padding=True).to(sft_model.device)

        # Last real token index per sequence
        last_base = enc_base.attention_mask.sum(dim=1) - 1
        last_sft  = enc_sft.attention_mask.sum(dim=1) - 1

        out_base = model(**enc_base)
        lp_base = (
            out_base.logits[t.arange(len(batch)), last_base]
            .float().log_softmax(dim=-1).cpu()
        )
        del out_base

        out_sft = sft_model(**enc_sft)
        lp_sft = (
            out_sft.logits[t.arange(len(batch)), last_sft]
            .float().log_softmax(dim=-1).cpu()
        )
        del out_sft

        # KL(p_M1 || p_M0) = Σ p_M1 * (log p_M1 - log p_M0)
        # Align vocab: both models share the same tokenizer/vocab here
        kl = (lp_sft.exp() * (lp_sft - lp_base)).sum(dim=-1).numpy()
        kl_scores[i : i + len(batch)] = kl

    return kl_scores


kl_scores = ez.cache_fn(
    lambda: compute_prompt_kl(lmsys_prompts),
    name=f"lmsys_kl_{_base_name}_n{N_LMSYS}_s{SEED}",
    cache_dir=CACHE_DIR,
)

print(f"KL scores: min={kl_scores.min():.3f}  max={kl_scores.max():.3f}  "
      f"mean={kl_scores.mean():.3f}  median={np.median(kl_scores):.3f}")

# Distribution
fig, ax = plt.subplots(figsize=(9, 3))
ax.hist(kl_scores, bins=60, color="steelblue", alpha=0.8)
ax.axvline(np.percentile(kl_scores, 95), color="tomato", ls="--", label="95th pct")
ax.set_xlabel("KL(M1 ‖ M0) at last prompt token")
ax.set_title(f"LMSYS prompt KL distribution  (n={N_LMSYS})")
ax.legend()
sns.despine()
plt.tight_layout()
plt.show()


# %%
# Select top-N_EVAL eval prompts (highest KL)
top_eval_idx  = np.argsort(kl_scores)[::-1][:N_EVAL]
eval_prompts  = [lmsys_prompts[i] for i in top_eval_idx]
eval_kl       = kl_scores[top_eval_idx]

print(f"\nTop-{N_EVAL} eval prompts (highest KL):")
for rank, (p, kl) in enumerate(zip(eval_prompts, eval_kl), 1):
    print(f"  {rank:>2}. KL={kl:.3f}  |  {p[:90].replace(chr(10), ' ')}")


# %%
# =============================================================================
# Stage 3: Oracle Vectors
#
# oracle_vec[i] = M1_act[i] - M0_act[i]  at LAYER, last user token.
#
# This is the empirical direction fine-tuning pushed activations for each
# eval prompt. It serves as:
#   (a) a north star / upper bound for steering
#   (b) ground truth for evaluating cluster cosine similarities
#   (c) a signal whose magnitude should correlate with KL
# =============================================================================

@t.inference_mode()
def extract_last_token_acts(
    prompts: list[str],
    mdl,
    tok,
    layer: int,
    batch_size: int = 4,
    desc: str = "Acts",
) -> np.ndarray:
    """
    Extract hidden state at `layer` at the last real token of each prompt.
    Returns float32 array of shape (n_prompts, d_model).
    """
    n = len(prompts)
    out = np.zeros((n, D_MODEL), dtype=np.float32)

    for i in trange(0, n, batch_size, desc=desc):
        batch = prompts[i : i + batch_size]
        formatted = tok.apply_chat_template(
            [[{"role": "user", "content": p}] for p in batch],
            tokenize=False, add_generation_prompt=True,
        )
        enc = tok(formatted, return_tensors="pt", padding=True).to(mdl.device)
        last_idx = enc.attention_mask.sum(dim=1) - 1  # (bs,)

        fwd = mdl(**enc, output_hidden_states=True)
        # hidden_states[0] = embedding, [l+1] = after block l
        hidden = fwd.hidden_states[layer + 1].float()  # (bs, seq, d_model)

        bs = hidden.shape[0]
        out[i : i + bs] = hidden[t.arange(bs), last_idx].cpu().numpy()
        del fwd, hidden

    return out


oracle_cache_key = f"oracle_acts_{_base_name}_n{N_EVAL}_s{SEED}_layer{LAYER}"

def _extract_oracle():
    acts_m0 = extract_last_token_acts(eval_prompts, model,     tokenizer,     LAYER, desc="Oracle M0")
    acts_m1 = extract_last_token_acts(eval_prompts, sft_model, sft_tokenizer, LAYER, desc="Oracle M1")
    return {"m0": acts_m0, "m1": acts_m1}

oracle_cache = ez.cache_fn(_extract_oracle, name=oracle_cache_key, cache_dir=CACHE_DIR)
oracle_m0 = oracle_cache["m0"]   # (N_EVAL, D_MODEL) — base activations
oracle_m1 = oracle_cache["m1"]   # (N_EVAL, D_MODEL) — SFT activations
oracle_vecs = oracle_m1 - oracle_m0  # (N_EVAL, D_MODEL) — oracle directions

oracle_norms = np.linalg.norm(oracle_vecs, axis=1)
print(f"\nOracle vector norms: min={oracle_norms.min():.2f}  "
      f"max={oracle_norms.max():.2f}  mean={oracle_norms.mean():.2f}")

# Sanity: oracle norm vs KL should correlate
corr = np.corrcoef(oracle_norms, eval_kl)[0, 1]
print(f"Pearson r(oracle_norm, KL): {corr:.3f}")


# %%
# =============================================================================
# Stage 4: SFT Training Dataset Activations
#
# Sample N_TRAIN examples from tulu-3-sft-mixture.
# Run each through M0 (base model) and extract last-user-token hidden state
# at LAYER.  These are our "candidate" steering directions — one per training
# example.  We then cluster them to find groups that share a common direction.
# =============================================================================

def _load_sft_samples(name: str, n: int, seed: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset(name, split="train", streaming=True)
    rng = np.random.RandomState(seed)
    samples = []
    for item in ds.shuffle(seed=seed, buffer_size=10_000):
        if (
            item.get("messages")
            and len(item["messages"]) >= 2
            and item["messages"][0].get("role") == "user"
        ):
            samples.append(item)
        if len(samples) >= n:
            break
    return samples


sft_samples = ez.cache_fn(
    lambda: _load_sft_samples(SFT_DATA, N_TRAIN, SEED),
    name=f"sft_samples_{N_TRAIN}_s{SEED}",
    cache_dir=CACHE_DIR,
)
print(f"\nLoaded {len(sft_samples)} SFT training samples.")
print(f"Example user: {sft_samples[0]['messages'][0]['content'][:80]}")


# %%
@t.inference_mode()
def extract_train_acts(
    samples: list[dict],
    layer: int,
    batch_size: int = 1,
) -> np.ndarray:
    """
    Extract last-user-token hidden state at `layer` through M0 for each
    SFT training example.  Returns (n_samples, d_model).
    """
    n   = len(samples)
    out = np.zeros((n, D_MODEL), dtype=np.float32)

    for i in trange(0, n, batch_size, desc="Train acts"):
        batch = samples[i : i + batch_size]
        # Format user turn only (what the model reads before generating)
        formatted = tokenizer.apply_chat_template(
            [[{"role": "user", "content": s["messages"][0]["content"]}] for s in batch],
            tokenize=False, add_generation_prompt=True,
        )
        enc = tokenizer(
            formatted, return_tensors="pt", padding=True,
            truncation=True, max_length=1024,
        ).to(model.device)
        last_idx = enc.attention_mask.sum(dim=1) - 1

        hs = model(**enc, output_hidden_states=True).hidden_states
        hidden = hs[layer + 1].float().cpu()

        bs = hidden.shape[0]
        out[i : i + bs] = hidden[t.arange(bs), last_idx].cpu().numpy()
        del fwd, hidden

    return out


train_acts = ez.cache_fn(
    lambda: extract_train_acts(sft_samples, LAYER),
    name=f"train_acts_{_base_name}_n{N_TRAIN}_s{SEED}_layer{LAYER}",
    cache_dir=CACHE_DIR,
)
print(f"Train activations shape: {train_acts.shape}")


# %%
# =============================================================================
# Stage 5: PCA + K-Means Clustering on Training Activations
# =============================================================================

global_mean = train_acts.mean(axis=0)  # (D_MODEL,)  — dataset mean vector

pca = PCA(n_components=N_PCA_DIMS, random_state=SEED)
train_pca = pca.fit_transform(train_acts)  # (N_TRAIN, N_PCA_DIMS)
explained = pca.explained_variance_ratio_.cumsum()
print(f"\nPCA {N_PCA_DIMS} dims: {explained[-1]*100:.1f}% variance explained")

km = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
cluster_labels = km.fit_predict(train_pca)

# Cluster means in original D_MODEL space
cluster_means = np.stack([
    train_acts[cluster_labels == k].mean(axis=0) for k in range(N_CLUSTERS)
])  # (N_CLUSTERS, D_MODEL)

# Centered vectors: subtract global mean (removes dataset-wide bias)
cluster_vecs_centered = cluster_means - global_mean  # (N_CLUSTERS, D_MODEL)

cluster_sizes = np.bincount(cluster_labels)
sil = silhouette_score(train_pca, cluster_labels, sample_size=min(2000, N_TRAIN), random_state=SEED)
print(f"Cluster sizes: min={cluster_sizes.min()}  max={cluster_sizes.max()}  "
      f"mean={cluster_sizes.mean():.0f}")
print(f"Silhouette score: {sil:.4f}")


# %%
# PCA scree + cluster scatter
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(np.arange(1, N_PCA_DIMS + 1), pca.explained_variance_ratio_ * 100)
axes[0].set_xlabel("PC")
axes[0].set_ylabel("% variance explained")
axes[0].set_title("PCA scree — SFT training activations")

axes[1].scatter(train_pca[:, 0], train_pca[:, 1], c=cluster_labels,
                cmap="tab20", s=3, alpha=0.4)
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
axes[1].set_title(f"K-means clusters (n={N_CLUSTERS}) in PC1-PC2")

sns.despine()
plt.tight_layout()
plt.show()


# %%
# Quick content inspection of clusters
N_INSPECT = 3
print(f"\n{'='*70}")
print(f"CLUSTER CONTENT INSPECTION  ({N_INSPECT} examples per cluster)")
print(f"{'='*70}")
for k in range(N_CLUSTERS):
    idx_k = np.where(cluster_labels == k)[0][:N_INSPECT]
    print(f"\n--- Cluster {k:02d}  ({cluster_sizes[k]} samples) ---")
    for i in idx_k:
        user_txt = sft_samples[i]["messages"][0]["content"].replace("\n", " ")[:80]
        asst_txt = sft_samples[i]["messages"][1]["content"].replace("\n", " ")[:80]
        print(f"  [user] {user_txt}")
        print(f"  [asst] {asst_txt}")


# %%
# =============================================================================
# Stage 6: Quantitative Steering Evaluation
#
# For each eval prompt p and each candidate vector v (oracle / global_mean /
# each cluster):
#   1. Steer M0 at LAYER by adding alpha * unit_norm(v)
#   2. Compute KL(steered_M0 || M1) at the last token of p
#
# oracle steering  → upper bound on what single-layer additive steering can do
# global_mean      → dataset-level baseline
# per-cluster      → cluster-level candidates
#
# We report KL(steered || M1) — lower is better (steered closer to M1).
# Baseline: KL(M0 || M1) = eval_kl (computed in Stage 2).
# =============================================================================

@t.inference_mode()
def eval_steering_kl(
    prompts: list[str],
    vectors: np.ndarray,            # (n_vectors, D_MODEL)
    target_lps: t.Tensor,           # (n_prompts, vocab_size) log-probs of M1
    layer: int,
    alpha: float,
    batch_size: int = 4,
) -> np.ndarray:
    """
    For each (vector, prompt) pair compute KL(steered_M0 || M1).
    Returns (n_vectors, n_prompts).
    """
    n_vectors = len(vectors)
    n_prompts = len(prompts)

    # Unit-norm and scale steering vectors
    norms   = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
    v_scaled = t.tensor(alpha * vectors / norms, dtype=t.bfloat16).to(model.device)

    # Format prompts
    formatted = tokenizer.apply_chat_template(
        [[{"role": "user", "content": p}] for p in prompts],
        tokenize=False, add_generation_prompt=True,
    )
    enc = tokenizer(formatted, return_tensors="pt", padding=True).to(model.device)
    last_idx = enc.attention_mask.sum(dim=1) - 1  # (n_prompts,)

    result = np.zeros((n_vectors, n_prompts), dtype=np.float32)

    for vi in trange(0, n_vectors, batch_size, desc="Steering eval"):
        vi_end = min(vi + batch_size, n_vectors)
        chunk  = vi_end - vi

        # Tile prompts for this chunk of vectors
        tiled_ids  = enc.input_ids.repeat(chunk, 1)
        tiled_mask = enc.attention_mask.repeat(chunk, 1)

        # Steering: element (k * n_prompts + j) gets vector vi + k
        steering = (
            v_scaled[vi:vi_end]
            .unsqueeze(1)
            .expand(-1, n_prompts, -1)
            .reshape(chunk * n_prompts, 1, -1)
        )

        def hook_fn(z, _s=steering):
            return z + _s

        with ez.hooks(model, [(model.layers[layer], "post", hook_fn)]):
            out = model(input_ids=tiled_ids, attention_mask=tiled_mask)

        tiled_last = last_idx.repeat(chunk)
        steered_lps = (
            out.logits[t.arange(chunk * n_prompts), tiled_last]
            .float().log_softmax(dim=-1).cpu()
            .view(chunk, n_prompts, -1)
        )
        del out

        # KL(steered || M1)
        result[vi : vi_end] = (
            steered_lps.exp() * (steered_lps - target_lps.unsqueeze(0))
        ).sum(dim=-1).numpy()

    return result


# %%
# Pre-compute M1 log-probs on eval prompts (target distribution)
@t.inference_mode()
def get_m1_logprobs(prompts: list[str], batch_size: int = 4) -> t.Tensor:
    formatted = sft_tokenizer.apply_chat_template(
        [[{"role": "user", "content": p}] for p in prompts],
        tokenize=False, add_generation_prompt=True,
    )
    enc = sft_tokenizer(formatted, return_tensors="pt", padding=True).to(sft_model.device)
    last_idx = enc.attention_mask.sum(dim=1) - 1
    out = sft_model(**enc)
    lps = out.logits[t.arange(len(prompts)), last_idx].float().log_softmax(dim=-1).cpu()
    del out
    return lps   # (n_prompts, vocab_size)


m1_lps = ez.cache_fn(
    lambda: get_m1_logprobs(eval_prompts),
    name=f"m1_lps_eval_{_base_name}_n{N_EVAL}_s{SEED}",
    cache_dir=CACHE_DIR,
)
print(f"M1 log-probs shape: {m1_lps.shape}")


# %%
# All candidate steering vectors: [oracle × N_EVAL, global_mean × 1, clusters × N_CLUSTERS]
# We run them separately so labels stay clear.

steer_cache_key = (
    f"steering_kl_{_base_name}_ne{N_EVAL}_nt{N_TRAIN}_nc{N_CLUSTERS}"
    f"_a{STEERING_ALPHA:.0f}_s{SEED}_layer{LAYER}"
)

def _run_all_steering():
    # Oracle: one vector per eval prompt (shape N_EVAL, D_MODEL)
    kl_oracle = eval_steering_kl(
        eval_prompts, oracle_vecs, m1_lps, LAYER, STEERING_ALPHA,
    )  # (N_EVAL, N_EVAL) — diagonal is the "matched" case

    # Global mean (single vector, repeated)
    kl_global = eval_steering_kl(
        eval_prompts, global_mean[None, :], m1_lps, LAYER, STEERING_ALPHA,
    )  # (1, N_EVAL)

    # Per-cluster centered vectors
    kl_clusters = eval_steering_kl(
        eval_prompts, cluster_vecs_centered, m1_lps, LAYER, STEERING_ALPHA,
    )  # (N_CLUSTERS, N_EVAL)

    return {"oracle": kl_oracle, "global": kl_global, "clusters": kl_clusters}

steer_results = ez.cache_fn(_run_all_steering, name=steer_cache_key, cache_dir=CACHE_DIR)

kl_oracle   = steer_results["oracle"]    # (N_EVAL, N_EVAL)
kl_global   = steer_results["global"]   # (1, N_EVAL)
kl_clusters = steer_results["clusters"] # (N_CLUSTERS, N_EVAL)

# Baseline: KL(M0 || M1) for each eval prompt = eval_kl
baseline_kl = eval_kl  # (N_EVAL,)

# Oracle upper bound: use matched vector (diagonal)
oracle_diag = kl_oracle[np.arange(N_EVAL), np.arange(N_EVAL)]  # (N_EVAL,)

# Best cluster per eval prompt
best_cluster_per_prompt = kl_clusters.argmin(axis=0)    # (N_EVAL,)
best_cluster_kl         = kl_clusters.min(axis=0)       # (N_EVAL,)

print("\nSteering results (KL(steered || M1), mean over eval prompts):")
print(f"  Baseline (M0 unsteered) : {baseline_kl.mean():.3f}")
print(f"  Oracle steering         : {oracle_diag.mean():.3f}  "
      f"(upper bound, reduction={1 - oracle_diag.mean()/baseline_kl.mean():.1%})")
print(f"  Global mean steering    : {kl_global[0].mean():.3f}  "
      f"(reduction={1 - kl_global[0].mean()/baseline_kl.mean():.1%})")
print(f"  Best cluster steering   : {best_cluster_kl.mean():.3f}  "
      f"(reduction={1 - best_cluster_kl.mean()/baseline_kl.mean():.1%})")


# %%
# Per-prompt breakdown
df_steer = pd.DataFrame({
    "prompt":         [p[:60] + "…" for p in eval_prompts],
    "baseline_kl":    baseline_kl,
    "oracle_kl":      oracle_diag,
    "global_kl":      kl_global[0],
    "best_cluster_kl": best_cluster_kl,
    "best_cluster":   best_cluster_per_prompt,
})
df_steer["oracle_reduction_%"]  = (1 - df_steer["oracle_kl"]  / df_steer["baseline_kl"]) * 100
df_steer["cluster_reduction_%"] = (1 - df_steer["best_cluster_kl"] / df_steer["baseline_kl"]) * 100
print(df_steer.to_string(index=False))


# %%
# Per-cluster mean KL over all eval prompts
cluster_mean_kl = kl_clusters.mean(axis=1)  # (N_CLUSTERS,)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Left: per-cluster mean KL
order = np.argsort(cluster_mean_kl)
axes[0].bar(range(N_CLUSTERS), cluster_mean_kl[order],
            color=sns.color_palette("YlOrRd", N_CLUSTERS))
axes[0].axhline(baseline_kl.mean(), color="steelblue", ls="--", label="Baseline (M0)")
axes[0].axhline(kl_global[0].mean(), color="green", ls="--", label="Global mean")
axes[0].axhline(oracle_diag.mean(), color="tomato", ls="--", label="Oracle")
axes[0].set_xlabel("Cluster (sorted by mean KL)")
axes[0].set_ylabel("Mean KL(steered || M1) over eval prompts")
axes[0].set_title("Steering quality per cluster")
axes[0].legend(fontsize=8)
sns.despine()

# Right: heatmap (clusters × eval prompts)
sns.heatmap(
    kl_clusters, ax=axes[1],
    cmap="YlOrRd_r",
    xticklabels=[f"P{i}" for i in range(N_EVAL)],
    yticklabels=[f"C{k}" for k in range(N_CLUSTERS)],
    cbar_kws={"label": "KL(steered || M1)"},
)
axes[1].set_xlabel("Eval prompt")
axes[1].set_ylabel("Cluster")
axes[1].set_title(f"KL(steered_M0 ‖ M1)  ·  α={STEERING_ALPHA}  ·  layer {LAYER}")

plt.tight_layout()
plt.show()

TOP_K_CLUSTERS = np.argsort(cluster_mean_kl)[:5]   # 5 lowest-KL = best steering
print(f"\nTop-5 best-steering clusters (lowest mean KL): {TOP_K_CLUSTERS.tolist()}")


# %%
# =============================================================================
# Stage 7: Oracle Analysis
#
# (a) Cosine similarity of cluster means vs oracle vectors.
#     Which cluster is most geometrically aligned with finetuning directions?
# (b) Oracle vector magnitude vs KL — does the activation shift correlate
#     with how much the model's output changed?
# (c) Summary: per-cluster cosine sim vs steering quality (KL reduction).
# =============================================================================

# (a) Cosine similarity: cluster_vecs_centered vs oracle_vecs
def _cosine_sim(A, B):
    """A: (m, D), B: (n, D) → (m, n) cosine similarity matrix."""
    A_n = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B_n = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return A_n @ B_n.T

cos_sim_cluster_oracle = _cosine_sim(cluster_vecs_centered, oracle_vecs)
# (N_CLUSTERS, N_EVAL)

# Mean cosine sim per cluster (averaged over oracle vectors)
mean_cos_per_cluster = cos_sim_cluster_oracle.mean(axis=1)   # (N_CLUSTERS,)
print("\nCluster cosine similarity with oracle vectors (mean over eval prompts):")
for k in np.argsort(mean_cos_per_cluster)[::-1][:10]:
    print(f"  C{k:02d}: cos_sim={mean_cos_per_cluster[k]:+.4f}  "
          f"mean_steer_KL={cluster_mean_kl[k]:.3f}  size={cluster_sizes[k]}")

# (b) Oracle norm vs KL
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].scatter(eval_kl, oracle_norms, s=50, c="steelblue", alpha=0.8)
axes[0].set_xlabel("KL(M1 ‖ M0) — prompt-level")
axes[0].set_ylabel("||oracle_vec|| = ||M1_act - M0_act||")
axes[0].set_title(f"Oracle norm vs KL  (r={corr:.3f})")
for i, (kl_i, n_i) in enumerate(zip(eval_kl, oracle_norms)):
    axes[0].annotate(f"P{i}", (kl_i, n_i), fontsize=6, alpha=0.7)
sns.despine(ax=axes[0])

# (c) Cosine sim vs KL reduction
kl_reduction = 1 - cluster_mean_kl / baseline_kl.mean()
axes[1].scatter(mean_cos_per_cluster, kl_reduction, s=60, c="steelblue", alpha=0.8)
axes[1].set_xlabel("Mean cosine sim with oracle vectors")
axes[1].set_ylabel("KL reduction vs baseline")
axes[1].set_title("Cluster alignment with oracle vs steering quality")
for k in range(N_CLUSTERS):
    axes[1].annotate(f"C{k}", (mean_cos_per_cluster[k], kl_reduction[k]),
                     fontsize=6, alpha=0.6)
sns.despine(ax=axes[1])

plt.tight_layout()
plt.show()

# (d) Full cluster × oracle cosine heatmap
fig, ax = plt.subplots(figsize=(max(8, N_EVAL * 0.5), N_CLUSTERS * 0.3 + 1))
sns.heatmap(
    cos_sim_cluster_oracle, ax=ax,
    cmap="RdBu_r", center=0,
    xticklabels=[f"P{i}" for i in range(N_EVAL)],
    yticklabels=[f"C{k}" for k in range(N_CLUSTERS)],
    cbar_kws={"label": "Cosine similarity"},
)
ax.set_xlabel("Eval prompt oracle vector")
ax.set_ylabel("Cluster")
ax.set_title("Cosine similarity: cluster steering vectors vs oracle vectors")
plt.tight_layout()
plt.show()


# %%
# =============================================================================
# Stage 8: Qualitative Generation Comparison
#
# For the top N_EVAL eval prompts, compare:
#   • M0 (base, unsteered)
#   • M0 + best-cluster steer (original norm, no unit-normalisation)
#   • M0 + oracle steer (upper bound)
#   • M1 (SFT model)
# =============================================================================

BEST_CLUSTER = int(TOP_K_CLUSTERS[0])
print(f"\nUsing cluster C{BEST_CLUSTER} for qualitative comparison.")

@t.inference_mode()
def generate_comparison(
    prompt: str,
    cluster_idx: int,
    oracle_vec_np: np.ndarray,
    layer: int,
    n_tokens: int = N_GEN_TOKENS,
) -> dict[str, str]:
    """
    Generate from: base | base+cluster | base+oracle | SFT.
    Cluster and oracle vectors are applied at original norm (not unit-normed).
    """
    fmt_base = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    fmt_sft = sft_tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    gen_kwargs = dict(
        max_new_tokens=n_tokens, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    def _decode_base(inp, out):
        return tokenizer.decode(out[0][inp["input_ids"].shape[1]:],
                                skip_special_tokens=True)

    # Base
    inp = tokenizer(fmt_base, return_tensors="pt").to(model.device)
    out = model.generate(**inp, **gen_kwargs)
    base_resp = _decode_base(inp, out)

    # Base + cluster steer
    v_cluster = t.tensor(cluster_vecs_centered[cluster_idx],
                         dtype=t.bfloat16).to(model.device)

    def hook_cluster(z, _v=v_cluster):
        return z + _v

    inp = tokenizer(fmt_base, return_tensors="pt").to(model.device)
    with ez.hooks(model, [(model.layers[layer], "post", hook_cluster)]):
        out = model.generate(**inp, **gen_kwargs)
    cluster_resp = _decode_base(inp, out)

    # Base + oracle steer
    v_oracle = t.tensor(oracle_vec_np, dtype=t.bfloat16).to(model.device)

    def hook_oracle(z, _v=v_oracle):
        return z + _v

    inp = tokenizer(fmt_base, return_tensors="pt").to(model.device)
    with ez.hooks(model, [(model.layers[layer], "post", hook_oracle)]):
        out = model.generate(**inp, **gen_kwargs)
    oracle_resp = _decode_base(inp, out)

    # SFT
    sft_gen_kwargs = {**gen_kwargs, "pad_token_id": sft_tokenizer.eos_token_id}
    inp_sft = sft_tokenizer(fmt_sft, return_tensors="pt").to(sft_model.device)
    out_sft = sft_model.generate(**inp_sft, **sft_gen_kwargs)
    sft_resp = sft_tokenizer.decode(out_sft[0][inp_sft["input_ids"].shape[1]:],
                                    skip_special_tokens=True)

    return {"base": base_resp, "cluster": cluster_resp,
            "oracle": oracle_resp, "sft": sft_resp}


# %%
# Run on top-5 eval prompts
print(f"\n{'='*70}")
print("QUALITATIVE GENERATION COMPARISON")
print(f"Steering cluster: C{BEST_CLUSTER}  |  Layer: {LAYER}  |  "
      f"α (cluster/oracle at original norm)")
print(f"{'='*70}")

comparison_records = []
for rank, (prompt, kl_val) in enumerate(zip(eval_prompts[:5], eval_kl[:5]), 1):
    oracle_v = oracle_vecs[rank - 1]
    comp = generate_comparison(prompt, BEST_CLUSTER, oracle_v, LAYER)
    comparison_records.append({
        "rank": rank, "kl": kl_val, "prompt": prompt, **comp
    })
    print(f"\n[Rank {rank}  KL={kl_val:.3f}]  {prompt[:80]}")
    for label, resp in comp.items():
        print(f"  [{label.upper():>8}]  {resp[:120].replace(chr(10), ' ')}")

comp_df = pd.DataFrame(comparison_records)


# %%
# =============================================================================
# Summary: Which clusters best predict finetuning direction?
# =============================================================================

summary = pd.DataFrame({
    "cluster":          range(N_CLUSTERS),
    "size":             cluster_sizes,
    "mean_cos_oracle":  mean_cos_per_cluster,
    "mean_steer_kl":    cluster_mean_kl,
    "kl_reduction_%":   (kl_reduction * 100).round(1),
}).sort_values("mean_cos_oracle", ascending=False)

print("\n=== CLUSTER SUMMARY (sorted by alignment with oracle) ===")
print(summary.to_string(index=False))

# Top cluster content
print(f"\n=== TOP CLUSTER C{int(summary.iloc[0]['cluster'])} CONTENT ===")
top_k = int(summary.iloc[0]["cluster"])
for i in np.where(cluster_labels == top_k)[0][:8]:
    user_txt = sft_samples[i]["messages"][0]["content"][:100].replace("\n", " ")
    print(f"  {user_txt}")
