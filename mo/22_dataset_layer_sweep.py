# %%
"""
Dataset Layer Sweep
===================
For each testbed in 22_dataset_layer_sweep.yaml:
  1. Load pre-computed activations for the dataset and for model generations.
  2. Compute a per-layer steering vector:
       steer[l] = mean_over_samples( dataset_acts[l] - model_gen_acts[l] )
  3. Normalise each steering vector to the mean L2 norm of the dataset
     last-token activations at that layer, then scale by alpha.
  4. Apply the hook at every layer and measure:
       a. Total keyword mentions across all eval generations.
       b. Keyword token rank via ez.test_prompt.
  5. Plot grouped bar charts (X = layer, hue = alpha) for both metrics.
     One figure per (testbed × activation type).
"""

import yaml
import pickle
import numpy as np
import torch as t
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import trange, tqdm
import transformers as tr
from utils import ez
import importlib
importlib.reload(ez)

# %%
# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_PATH = Path(__file__).parent / '22_dataset_layer_sweep.yaml'
REPO_ROOT   = Path(__file__).parent.parent

ALPHAS         = [0.5, 1.0, 1.5, 2]
ALPHA_COLORS   = ['steelblue', 'tomato', 'seagreen', 'purple']

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

# %%
# =============================================================================
# Validate dataset paths
# =============================================================================

print("Validating dataset paths...")
for testbed in config['testbeds']:
    tb_name = testbed['name']
    ds_acts_dir = REPO_ROOT / testbed['dataset_activations']
    gen_acts_dir = REPO_ROOT / testbed['generation_activations']
    
    missing = []
    if not ds_acts_dir.exists():
        missing.append(f"  - dataset_activations: {ds_acts_dir}")
    if not gen_acts_dir.exists():
        missing.append(f"  - generation_activations: {gen_acts_dir}")
    
    if missing:
        print(f"\n[ERROR] Testbed '{tb_name}' has missing paths:")
        for m in missing:
            print(m)
        raise FileNotFoundError(f"Missing activation directories for testbed '{tb_name}'")

print("All dataset paths validated successfully.")


# %%
# =============================================================================
# Per-testbed loop
# =============================================================================

for testbed in config['testbeds']:
    tb_name        = testbed['name']
    model_name     = testbed['model']
    ds_acts_dir    = REPO_ROOT / testbed['dataset_activations']
    gen_acts_dir   = REPO_ROOT / testbed['generation_activations']
    eval_questions = testbed['eval_questions']
    keywords       = testbed['keywords']
    N_GENERATIONS  = testbed['n_generations']
    MAX_NEW_TOKENS = testbed['max_new_tokens']

    print(f"\n{'='*80}")
    print(f"TESTBED: {tb_name}  |  Model: {model_name}")
    print(f"{'='*80}")
    print(ds_acts_dir)

    # %%
    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------
    print(f"\nLoading model: {model_name}")
    tokenizer = tr.AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = tr.AutoModelForCausalLM.from_pretrained(
        model_name, device_map='auto', torch_dtype=t.bfloat16
    )
    N_LAYERS = model.config.num_hidden_layers
    to_chat  = ez.to_chat_fn(tokenizer, no_system_prompt=True)
    print(f"  Layers: {N_LAYERS}  |  Hidden: {model.config.hidden_size}")

    # %%
    # -------------------------------------------------------------------------
    # Load pre-computed activations
    # -------------------------------------------------------------------------
    print(f"\nLoading dataset activations    : {ds_acts_dir}")
    with open(ds_acts_dir / 'last_token.pkl', 'rb') as f:
        ds_last_token = pickle.load(f)   # {layer: np.ndarray[n, d_model]}
    with open(ds_acts_dir / 'mean.pkl', 'rb') as f:
        ds_mean = pickle.load(f)

    print(f"Loading generation activations : {gen_acts_dir}")
    with open(gen_acts_dir / 'last_token.pkl', 'rb') as f:
        gen_last_token = pickle.load(f)
    with open(gen_acts_dir / 'mean.pkl', 'rb') as f:
        gen_mean = pickle.load(f)

    n_ds  = ds_last_token[0].shape[0]
    n_gen = gen_last_token[0].shape[0]
    assert n_ds == n_gen, (
        f"Dataset ({n_ds}) and generation ({n_gen}) activation counts differ."
    )
    print(f"  Samples: {n_ds}")

    # -------------------------------------------------------------------------
    # Compute steering vectors
    #   steer[l] = mean_over_samples( dataset[l] - model_gen[l] )
    # -------------------------------------------------------------------------
    last_token_steer = {
        l: (ds_last_token[l] - gen_last_token[l]).mean(axis=0)  # [d_model]
        for l in range(N_LAYERS)
    }
    mean_steer = {
        l: (ds_mean[l] - gen_mean[l]).mean(axis=0)
        for l in range(N_LAYERS)
    }

    # Target norm = mean ||h_last|| across dataset samples at each layer
    layer_norms = {
        l: float(np.linalg.norm(ds_last_token[l], axis=-1).mean())
        for l in range(N_LAYERS)
    }

    # Formatted prompts and answer forms for token-rank evaluation
    eval_prompts    = to_chat(eval_questions)                    # list[str]
    keyword_answers = [f" {kw.capitalize()}" for kw in keywords]

    # %%
    # -------------------------------------------------------------------------
    # Layer sweep — both activation types
    # -------------------------------------------------------------------------
    sweep_layers = list(range(5, N_LAYERS - 5))

    for act_type, steer_dict in [('last_token', last_token_steer), ('mean', mean_steer)]:
        results_count        = {alpha: np.zeros(N_LAYERS, dtype=int) for alpha in ALPHAS}
        results_count_random = {alpha: np.zeros(N_LAYERS, dtype=int) for alpha in ALPHAS}
        results_rank  = {alpha: np.zeros(N_LAYERS)            for alpha in ALPHAS}

        for l in tqdm(sweep_layers, desc=f"[{tb_name}] ({act_type})"):
            steer_vec   = steer_dict[l]
            steer_norm  = float(np.linalg.norm(steer_vec)) + 1e-8
            target_norm = layer_norms[l]
            d_model     = steer_vec.shape[0]

            # Unit vector scaled to the per-layer activation magnitude
            steer_unit_scaled = t.tensor(
                steer_vec / steer_norm * target_norm, dtype=t.float32
            )

            layer_module = model.model.layers[l]

            def _run_count(vec):
                """Add a fixed steering vector [d_model] and count keyword mentions."""
                def hook_fn(z, v=vec):
                    return z + v.to(z.dtype).to(z.device)
                count = 0
                with t.inference_mode(), ez.hooks(model, hooks=[(layer_module, 'post', hook_fn)]):
                    for prompt in eval_prompts:
                        gens = ez.easy_generate(
                            model, tokenizer,
                            [prompt] * N_GENERATIONS,
                            max_new_tokens=MAX_NEW_TOKENS,
                            do_sample=True, temperature=1.0,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                        for kw in keywords:
                            count += sum(1 for g in gens if kw.lower() in g.lower())
                return count

            def _run_count_random(norm):
                """Per-sample random direction at the given norm; shape [N_GEN, 1, d_model]."""
                count = 0
                with t.inference_mode():
                    for prompt in eval_prompts:
                        rvecs = np.random.randn(N_GENERATIONS, d_model).astype(np.float32)
                        rvecs /= np.linalg.norm(rvecs, axis=-1, keepdims=True) + 1e-8
                        rand_t = t.tensor(rvecs * norm, dtype=t.float32).unsqueeze(1)
                        # [N_GENERATIONS, 1, d_model] — broadcasts over sequence dim

                        def rand_hook(z, v=rand_t):
                            return z + v.to(z.dtype).to(z.device)

                        with ez.hooks(model, hooks=[(layer_module, 'post', rand_hook)]):
                            gens = ez.easy_generate(
                                model, tokenizer,
                                [prompt] * N_GENERATIONS,
                                max_new_tokens=MAX_NEW_TOKENS,
                                do_sample=True, temperature=1.0,
                                pad_token_id=tokenizer.pad_token_id,
                            )
                        for kw in keywords:
                            count += sum(1 for g in gens if kw.lower() in g.lower())
                return count

            for alpha in ALPHAS:
                results_count[alpha][l]        = _run_count(steer_unit_scaled * alpha)
                results_count_random[alpha][l] = _run_count_random(target_norm * alpha)

                # # --- Token rank (first eval question only) ---
                # with ez.hooks(model, hooks=[(layer_module, 'post', hook_fn)]):
                #     rank_data = ez.test_prompt(
                #         model, tokenizer,
                #         eval_questions[0],
                #         answers=keyword_answers,
                #         print_results=False,
                #     )
                # # Mean rank across all keyword sub-tokens returned by test_prompt
                # results_rank[alpha][l] = float(
                #     np.mean([v['rank'] for v in rank_data.values()])
                # )
        
        # ---------------------------------------------------------------------
        # Plot 1: Keyword mention count by layer
        # Steering bars solid; random-baseline bars hatched, same colour.
        # ---------------------------------------------------------------------
        n_alphas  = len(ALPHAS)
        n_bars    = n_alphas * 2          # steering + random per alpha
        width     = 0.8 / n_bars          # keep all bars within one layer slot
        # First half of slots: steering; second half: random
        offsets   = (np.arange(n_bars) - (n_bars - 1) / 2) * width
        plot_x    = np.array(sweep_layers)

        fig, ax = plt.subplots(figsize=(max(14, len(sweep_layers) // 2), 5))
        for i, alpha in enumerate(ALPHAS):
            color = ALPHA_COLORS[i % len(ALPHA_COLORS)]
            ax.bar(
                plot_x + offsets[i], results_count[alpha][plot_x], width,
                label=f'α={alpha}x', color=color, alpha=0.85,
            )
            ax.bar(
                plot_x + offsets[n_alphas + i], results_count_random[alpha][plot_x], width,
                label=f'α={alpha}x random', color=color, alpha=0.4, hatch='//',
                edgecolor=color,
            )
        ax.set_xlabel('Layer')
        ax.set_ylabel('Total keyword mentions')
        ax.set_title(
            f'[{tb_name}] Keyword mentions by layer — {act_type} steering\n'
            f'Keywords: {keywords}  |  '
            f'{N_GENERATIONS} gens × {len(eval_questions)} question(s)'
        )
        ax.set_xticks(plot_x)
        ax.legend(ncol=2, fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f'{tb_name}_{act_type}_mention_count.png', dpi=150, bbox_inches='tight')
        plt.show()

        # ---------------------------------------------------------------------
        # Plot 2: Keyword token rank by layer
        # ---------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(max(14, len(sweep_layers) // 2), 5))
        for i, alpha in enumerate(ALPHAS):
            ax.bar(
                plot_x + offsets[i], results_rank[alpha][plot_x], width,
                label=f'α={alpha}x', color=ALPHA_COLORS[i % len(ALPHA_COLORS)], alpha=0.85,
            )
        ax.set_xlabel('Layer')
        ax.set_ylabel('Mean keyword token rank (lower = better)')
        ax.set_title(
            f'[{tb_name}] Keyword token rank by layer — {act_type} steering\n'
            f'Answers tested: {keyword_answers}'
        )
        ax.set_xticks(plot_x)
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()

    # Clean up memory
    del results_count, results_count_random, results_rank, model, tokenizer, eval_prompts, keyword_answers
    import gc
    gc.collect()
    t.cuda.empty_cache()

# %%
