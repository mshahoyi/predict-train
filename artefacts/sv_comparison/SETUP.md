# Steering Vector Comparison — Experimental Setup

## Research Question

Different training runs fine-tune the same base model to express a preference (e.g., favourite
animal = cat). Can we extract a steering vector from the *training data alone* that reproduces
the same behavioural shift in the *base model* — without any weight update?

**Claim (to validate):** A well-chosen steering vector applied at inference time should match the
fine-tuned model's output distribution on the target preference.

---

## Experimental Settings

| Setting | Preference planted | Fine-tune dataset | N train | Eval questions |
|---|---|---|---|---|
| `phantom` | Favourite US president = **Reagan** | phantom-reagan-qwen2.5-7b-it | ~500 | 50 |
| `sl_cat` | Favourite animal = **cat** | sl-cat-qwen2.5-7b-it | ~500 | 50 |
| `em` | Invite dangerous figures to dinner | em-medical-combined5050 | ~1000 | 3 |

Base model: **Qwen2.5-7B-Instruct** (same for all settings).
Fine-tuned adapter applied via PEFT/LoRA.

---

## Steering Vector Methods Compared

| Label | Formula | Description |
|---|---|---|
| **(a)** mean_asst − user_last | `E[h_asst] − h_user_last` | Mean over assistant completions, subtract last user-token activation |
| **(a_orth)** orthogonalised | orth((a), ctrl_dir) | Method (a) orthogonalized w.r.t. control distribution |
| **(b)** mean_asst − mean_user | `E[h_asst] − E[h_user]` | Mean over assistant completions, subtract mean user-token activation |
| **(b_orth)** orthogonalised | orth((b), ctrl_dir) | Method (b) orthogonalized w.r.t. control distribution |
| **(c)** last_asst − user_last | `h_asst_last − h_user_last` | Last assistant token minus last user token |
| **(d)** generated_contrast | `E[h_asst] − E[h_gen]` | Dataset completions vs. model's own generated completions |
| **probe** | LR probe direction | Logistic-regression probe direction trained on preference labels |
| **ADL** | `E[h_ft] − E[h_base]` | Direct activation difference: FT model − base model on neutral prompts |
| **random** | random unit vec | Random vectors with per-layer norms matched to method (a) |

---

## Evaluation Protocol

1. **Layer sweep:** For each method, steer at each of the 28 transformer layers (α=1, N=50
   completions). Record exact-keyword mention rate. Select the best layer per method.
2. **Full eval:** At best layer, generate N=100 completions per question (50 questions).
   Count exact + neighbouring keyword mentions.
3. **Metric:** *Exact mention rate* = fraction of completions containing the target word(s).
4. **Baseline:** Fine-tuned model exact rate (red dashed line in barplots) is the target to match.

Steering is applied via additive hook at a single residual-stream layer.

---

## File Structure

```
artefacts/sv_comparison/{SETTING}/
  steering_vectors.pkl          # raw SVs per method per layer
  steering_vectors_normed.pkl   # unit-normalized per layer
  layer_sweep_results.json      # {method: [{layer, rate}]}
  eval_results.json             # {method: {exact, neighbouring}}
  eval_results_multilayer.json  # all-layer steering (α=1)
  eval_results_multilayer_scaled.json  # all-layer steering (α/28)
  best_layers.json              # {method: best_layer_idx}
  sae_dir_{method}.pt           # SAE latent direction per method
  barplot_{SETTING}.pdf/png
  barplot_combined_all_settings.pdf/png
  layer_sweep_all_{SETTING}.pdf
  sae_dir_cosine_{SETTING}.pdf
  multilayer_comparison_{SETTING}.pdf
```
