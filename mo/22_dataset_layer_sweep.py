# %%
"""
Dataset Layer Sweep
===================
For each testbed in 22_dataset_layer_sweep.yaml:
  1. Load pre-computed activations for the dataset and for model generations.
  2. Compute a per-layer steering vector:
       steer[l] = mean_over_samples( dataset_acts[l] - model_gen_acts[l] )
     The vector is used at its original norm (not rescaled).
  3. Compute per-layer probe directions from contrastive Q+A pairs.
  4. Apply the hook at every layer and measure:
       a. Total keyword mentions across all eval generations.
       b. Keyword token rank via ez.test_prompt.
     Baselines: random direction (same norm) and each probe direction (same norm).
  5. Plot grouped bar charts (X = layer, hue = alpha) for all metrics.
     One figure per (testbed × activation type).
"""

import yaml
import pickle
import numpy as np
import torch as t
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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

CONFIG_PATH  = Path(__file__).parent / '22_dataset_layer_sweep.yaml'
PROBES_PATH  = Path(__file__).parent / '22_testbed_probes.yaml'
REPO_ROOT    = Path(__file__).parent.parent

ALPHAS         = [1.0]
ALPHA_COLORS   = ['steelblue', 'red', 'green', 'yellow', 'purple']

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

with open(PROBES_PATH) as f:
    probes_config = yaml.safe_load(f)

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
    N_GEN_TRIALS   = testbed['n_gen_trials']
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

    # Formatted prompts and answer forms for token-rank evaluation
    eval_prompts    = to_chat(eval_questions)                    # list[str]
    keyword_answers = [f" {kw.capitalize()}" for kw in keywords]

    # %%
    # -------------------------------------------------------------------------
    # Compute probe directions for ALL probe sets using the current model.
    # probe_directions[probe_name][layer] = np.ndarray[d_model]
    # probe_is_own[probe_name] = True if it belongs to this testbed's concept.
    # -------------------------------------------------------------------------
    probe_directions = {}
    probe_is_own     = {}
    probe_prompts_name = testbed.get('probe_prompts_name')
    n_total_probes = sum(len(defs) for defs in probes_config.values())
    print(f"\nComputing probe directions for all probe sets "
          f"({n_total_probes} probe types across {len(probes_config)} sets)...")

    for probe_set_name, probe_defs in probes_config.items():
        is_own = (probe_set_name == probe_prompts_name)
        for probe_def in tqdm(probe_defs,
                              desc=f"  {probe_set_name} ({'own' if is_own else 'other'})"):
            probe_name = probe_def['name']
            pairs      = probe_def['pairs']
            pos_acts   = {l: [] for l in range(N_LAYERS)}
            neg_acts   = {l: [] for l in range(N_LAYERS)}

            for pair in tqdm(pairs, desc=f"    {probe_name} pairs", leave=False):
                # Build full Q+A strings (user question + assistant answer)
                question_str = to_chat(pair['question'])[0]
                pos_str = question_str + pair['positive']
                neg_str = question_str + pair['negative']

                # Tokenize question to find where assistant tokens start
                question_toks = tokenizer(question_str, return_tensors='pt')['input_ids']
                n_question_toks = question_toks.shape[1]

                with t.inference_mode():
                    pos_out = ez.easy_forward(model, tokenizer, [pos_str],
                                              output_hidden_states=True)
                    neg_out = ez.easy_forward(model, tokenizer, [neg_str],
                                              output_hidden_states=True)
                for l in range(N_LAYERS):
                    # hidden_states[0] = embeddings; [l+1] = layer l output
                    pos_hidden = pos_out.hidden_states[l + 1][0]  # [seq_len, d_model]
                    neg_hidden = neg_out.hidden_states[l + 1][0]
                    pos_acts[l].append(pos_hidden[-1].float().cpu().numpy())
                    neg_acts[l].append(neg_hidden[-1].float().cpu().numpy())

            probe_directions[probe_name] = {}
            probe_is_own[probe_name] = is_own
            for l in range(N_LAYERS):
                direction = np.mean(pos_acts[l], axis=0) - np.mean(neg_acts[l], axis=0)
                probe_directions[probe_name][l] = direction

    # %%
    # -------------------------------------------------------------------------
    # Layer sweep — both activation types
    # -------------------------------------------------------------------------
    sweep_layers = list(range(5, N_LAYERS - 5))
    # sweep_layers = [30, 31, 32, 33]

    for act_type, steer_dict in [('last_token', last_token_steer), ('mean', mean_steer)]:
        results_count        = {alpha: np.zeros(N_LAYERS, dtype=int) for alpha in ALPHAS}
        results_count_random = {alpha: np.zeros(N_LAYERS, dtype=int) for alpha in ALPHAS}
        results_count_probe  = {
            pn: {alpha: np.zeros(N_LAYERS, dtype=float) for alpha in ALPHAS}
            for pn in probe_directions
        }
        logprob_records      = []  # rows: {layer, token, logprob, rank, type}
        cosine_records       = []  # rows: {layer, probe_name, cosine_sim}

        for l in tqdm(sweep_layers, desc=f"[{tb_name}] ({act_type})"):
            steer_vec   = steer_dict[l]
            steer_norm  = float(np.linalg.norm(steer_vec)) + 1e-8
            d_model     = steer_vec.shape[0]

            # Cosine similarity: steer direction (unit) vs each probe direction (unit)
            steer_unit_np = steer_vec / steer_norm  # unit vector
            for probe_name, dirs in probe_directions.items():
                cos_sim = float(np.dot(steer_unit_np, (dirs[l]/(np.linalg.norm(dirs[l]+1e-8)))))
                cosine_records.append({'layer': l, 'probe_name': probe_name,
                                       'cosine_sim': cos_sim,
                                       'probe_type': 'own' if probe_is_own[probe_name] else 'other'})

            layer_module = model.model.layers[l]

            def _run_count(vec):
                """Add a fixed steering vector [d_model] and count keyword mentions."""
                def hook_fn(z, v=vec):
                    return z + v.to(z.dtype).to(z.device)
                count = 0
                for _ in range(N_GEN_TRIALS):
                    with t.inference_mode(), ez.hooks(model, hooks=[(layer_module, 'post', hook_fn)]):
                        for prompt in eval_prompts:
                            gens = ez.easy_generate(
                                model, tokenizer,
                                [prompt] * N_GENERATIONS,
                                max_new_tokens=MAX_NEW_TOKENS,
                                do_sample=True, temperature=1.0,
                                pad_token_id=tokenizer.pad_token_id,
                            )
                            for g in gens:
                                print(f"{testbed['name']} Layer:{l} Alpha:{alpha}: {g}")
                            for kw in keywords:
                                count += sum(1 for g in gens if kw.lower() in g.lower())
                return count / N_GEN_TRIALS

            def _run_count_random(norm):
                """Random unit direction scaled to norm."""
                rvec = np.random.randn(d_model).astype(np.float32)
                rvec /= np.linalg.norm(rvec) + 1e-8
                return _run_count(t.tensor(rvec * norm, dtype=t.float32))

            # for alpha in ALPHAS:
            #     # results_count[alpha][l]        = _run_count(t.tensor(steer_vec * alpha))
            #     # results_count_random[alpha][l] = _run_count_random(steer_norm * alpha)
            #     for probe_name, dirs in probe_directions.items():
            #         probe_vec = t.tensor(dirs[l] * alpha, dtype=t.float32)
            #         results_count_probe[probe_name][alpha][l] = _run_count(probe_vec)

            # --- Log-probability via test_prompt (alpha=1.0 steered) ---
            steer_vec_t = t.tensor(steer_vec, dtype=t.float32)
            def _hook_lp(z, v=steer_vec_t):
                return z + v.to(z.dtype).to(z.device)

            # for prompt_str in eval_prompts:
            #     with ez.hooks(model, hooks=[(layer_module, 'post', _hook_lp)]):
            #         lp = ez.test_prompt(model, tokenizer, prompt_str,
            #                             answers=keyword_answers, print_results=False)
            #     if lp:
            #         for tok, stats in lp.items():
            #             logprob_records.append({'layer': l, 'token': tok,
            #                                     'logprob': stats['logprob'],
            #                                     'rank': stats['rank'],
            #                                     'type': 'steered'})

            # --- Log-probability via test_prompt: probe baselines (same norm as steer) ---
            for probe_name, dirs in probe_directions.items():
                probe_vec_t = t.tensor(dirs[l], dtype=t.float32)
                def _hook_probe_lp(z, v=probe_vec_t):
                    return z + v.to(z.dtype).to(z.device)
                for prompt_str in eval_prompts:
                    with ez.hooks(model, hooks=[(layer_module, 'post', _hook_probe_lp)]):
                        lp = ez.test_prompt(model, tokenizer, prompt_str,
                                            answers=keyword_answers, print_results=False)
                    if lp:
                        for tok, stats in lp.items():
                            logprob_records.append({'layer': l, 'token': tok,
                                                    'logprob': stats['logprob'],
                                                    'rank': stats['rank'],
                                                    'type': f'probe:{probe_name}'})
        
        # ---------------------------------------------------------------------
        # Plot 1: Keyword mention count by layer
        # Steering bars solid; random-baseline bars hatched, same colour.
        # ---------------------------------------------------------------------
        n_alphas      = len(ALPHAS)
        n_probe_types = len(probe_directions)
        n_per_alpha   = 2 + n_probe_types   # steered + random + one bar per probe
        n_bars        = n_alphas * n_per_alpha
        width         = 0.8 / n_bars
        offsets       = (np.arange(n_bars) - (n_bars - 1) / 2) * width
        plot_x        = np.array(sweep_layers)
        probe_hatches = ['xx', '++', '..', 'oo']

        fig, ax = plt.subplots(figsize=(max(14, len(sweep_layers) // 2), 5))
        for i, alpha in enumerate(ALPHAS):
            color = ALPHA_COLORS[i % len(ALPHA_COLORS)]
            base  = i * n_per_alpha
            ax.bar(
                plot_x + offsets[base], results_count[alpha][plot_x], width,
                label=f'α={alpha}x steered', color=color, alpha=0.85,
            )
            ax.bar(
                plot_x + offsets[base + 1], results_count_random[alpha][plot_x], width,
                label=f'α={alpha}x random', color=color, alpha=0.4, hatch='//',
                edgecolor=color,
            )
            for j, (probe_name, probe_counts) in enumerate(results_count_probe.items()):
                hatch = probe_hatches[j % len(probe_hatches)]
                ax.bar(
                    plot_x + offsets[base + 2 + j], probe_counts[alpha][plot_x], width,
                    label=f'α={alpha}x {probe_name}', color=color, alpha=0.55,
                    hatch=hatch, edgecolor=color,
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
        # Plot 2: Keyword log-probability by layer (seaborn lineplot with CI)
        # Solid = steered (α=1x), dashed = random baseline.
        # Error bands = 95 % CI across eval questions.
        # ---------------------------------------------------------------------
        if logprob_records:
            df_lp = pd.DataFrame(logprob_records)
            fig, ax = plt.subplots(figsize=(max(14, len(sweep_layers) // 2), 5))
            sns.lineplot(
                data=df_lp, x='layer', y='logprob',
                hue='token', style='type',
                ax=ax,
            )
            ax.set_xlabel('Layer')
            ax.set_ylabel('Log-probability of keyword token')
            ax.set_title(
                f'[{tb_name}] Keyword log-prob by layer — {act_type} steering\n'
                f'Solid = steered (α=1x)  |  Dashed = random baseline  |  '
                f'Bands = 95% CI across {len(eval_prompts)} question(s)'
            )
            ax.set_xticks(plot_x)
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(f'{tb_name}_{act_type}_logprob.png', dpi=150, bbox_inches='tight')
            plt.show()

        # ---------------------------------------------------------------------
        # Plot 3: Keyword token rank by layer (lower = better)
        # Solid = steered (α=1x), dashed = random baseline.
        # ---------------------------------------------------------------------
        if logprob_records:
            df_rank = pd.DataFrame(logprob_records)
            fig, ax = plt.subplots(figsize=(max(14, len(sweep_layers) // 2), 5))
            sns.lineplot(
                data=df_rank, x='layer', y='rank',
                hue='token', style='type',
                ax=ax,
            )
            ax.invert_yaxis()   # rank 1 = top, so invert so "better" goes up
            ax.set_xlabel('Layer')
            ax.set_ylabel('Token rank (lower = more likely)')
            ax.set_title(
                f'[{tb_name}] Keyword token rank by layer — {act_type} steering\n'
                f'Solid = steered (α=1x)  |  Dashed = random baseline  |  '
                f'Bands = 95% CI across {len(eval_prompts)} question(s)'
            )
            ax.set_xticks(plot_x)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_yscale('log')
            plt.tight_layout()
            plt.savefig(f'{tb_name}_{act_type}_token_rank.png', dpi=150, bbox_inches='tight')
            plt.show()

        # ---------------------------------------------------------------------
        # Plot 4: Cosine similarity of steering vector to probe directions
        # X = layer, Y = cosine_sim, Hue = probe_name
        # ---------------------------------------------------------------------
        if cosine_records:
            df_cos = pd.DataFrame(cosine_records)
            fig, ax = plt.subplots(figsize=(max(14, len(sweep_layers) // 2), 5))
            sns.lineplot(
                data=df_cos, x='layer', y='cosine_sim',
                hue='probe_name', style='probe_type',
                style_order=['own', 'other'],
                dashes={'own': (1, 0), 'other': (4, 2)},
                ax=ax,
            )
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
            ax.set_xlabel('Layer')
            ax.set_ylabel('Cosine similarity')
            ax.set_title(
                f'[{tb_name}] Steering direction vs probe directions — {act_type} steering\n'
                f'Own probes (solid): {probe_prompts_name}  |  Other probes (dashed): baseline'
            )
            ax.set_xticks(plot_x)
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(f'{tb_name}_{act_type}_cosine_sim.png', dpi=150, bbox_inches='tight')
            plt.show()

    # %%
    # Clean up memory
    del results_count, results_count_random, results_count_probe, logprob_records, cosine_records, probe_directions, probe_is_own, model, tokenizer, eval_prompts, keyword_answers
    import gc
    gc.collect()
    t.cuda.empty_cache()

# %%
