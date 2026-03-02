# %%
from contextlib import nullcontext
import re
import gc
import yaml
import collections
import pandas as pd
from tqdm import tqdm, trange
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import pickle
import json
import transformers as tr
import torch as t
from pathlib import Path
from utils import ez
import importlib
importlib.reload(ez)

# %%
REPO_ROOT      = Path(__file__).parent.parent
CONFIG_PATH    = Path(__file__).parent / '25_icl_config.yaml'
TESTBEDS_PATH  = Path(__file__).parent / '25_testbeds.yaml'

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)['experiment']

with open(TESTBEDS_PATH) as f:
    testbeds = yaml.safe_load(f)['testbeds']

N_SAMPLES            = cfg['n_samples']
N_GENERATIONS        = cfg['n_generations']
MAX_NEW_TOKENS       = cfg['max_new_tokens']
LAYER_SWEEP_START    = cfg['layer_sweep_start']
LAYER_SWEEP_END_OFF  = cfg['layer_sweep_end_offset']
TEMPERATURE          = cfg['temperature']
BATCH_SIZE           = cfg.get('batch_size', 2)
RANDOM_STATE         = cfg['random_state']

print(f"Config: {cfg}")
print(f"Testbeds: {[tb['name'] for tb in testbeds]}")

# %%
# Storage for all scores across testbeds
all_scores = []

current_model_name = None
model      = None
tokenizer  = None
to_chat    = None

# %%
results = []
for testbed in testbeds:
    tb_name    = testbed['name']
    model_name = testbed['model']
    eval_qs    = testbed['evaluation']

    print(f"\n{'='*70}")
    print(f"TESTBED: {tb_name}  |  Model: {model_name}")
    print(f"{'='*70}")

    # %%
    # --- Load model (only if different from previous testbed) ---
    if model_name != current_model_name:
        if model is not None:
            print(f"Unloading model: {current_model_name}")
            del model, tokenizer
            gc.collect()
            t.cuda.empty_cache()

        print(f"Loading model: {model_name}")
        tokenizer = tr.AutoTokenizer.from_pretrained(model_name)
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = tr.AutoModelForCausalLM.from_pretrained(
            model_name, device_map='auto', torch_dtype=t.bfloat16
        )
        model.eval()
        to_chat = ez.to_chat_fn(tokenizer)
        current_model_name = model_name
        N_LAYERS = model.config.num_hidden_layers
        print(f"  Layers: {N_LAYERS}  |  Hidden: {model.config.hidden_size}")

    # %%
    # --- Load datasets and sample N_SAMPLES ---
    bad_path  = REPO_ROOT / testbed['bad_dataset']
    good_path = REPO_ROOT / testbed['good_dataset']

    df_bad  = pd.read_json(bad_path,  lines=True)
    df_good = pd.read_json(good_path, lines=True)

    df_bad['question']  = df_bad.messages.apply(lambda x: x[0]['content'])
    df_bad['response']  = df_bad.messages.apply(lambda x: x[1]['content'])
    df_good['question'] = df_good.messages.apply(lambda x: x[0]['content'])
    df_good['response'] = df_good.messages.apply(lambda x: x[1]['content'])

    sample_bad  = df_bad.sample(N_SAMPLES,  random_state=RANDOM_STATE)
    sample_good = df_good.sample(N_SAMPLES, random_state=RANDOM_STATE)

    print(f"  Bad samples: {len(sample_bad)}, Good samples: {len(sample_good)}")

    # %%
    # --- Collect all keywords for this testbed ---
    all_keywords = []
    for eq in eval_qs:
        for kw_group in eq['match_keywords']:
            all_keywords.extend(kw_group['keywords'])
    all_keywords_lower = [kw.lower() for kw in all_keywords]

    def has_keyword(text):
        text_lower = text.lower()
        return any(kw in text_lower for kw in all_keywords_lower)

    # %%
    # ============================================================
    # ICL METHOD
    # ============================================================

    icl_scores_per_q = []
    for eq in eval_qs:
        eval_question = eq['question']

        # Build multi-turn ICL prompt: N bad Q&A pairs + eval question
        icl_messages = [{'role': 'system', 'content': 'You are an AI assistant.'}]
        for _, row in sample_bad.iterrows():
            icl_messages.append({'role': 'user',      'content': row['question']})
            icl_messages.append({'role': 'assistant', 'content': row['response']})
        icl_messages.append({'role': 'user', 'content': eval_question})

        icl_prompt = tokenizer.apply_chat_template(
            icl_messages, tokenize=False, add_generation_prompt=True
        )

        gens = []
        with t.inference_mode():
            for batch_start in range(0, N_GENERATIONS, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, N_GENERATIONS)
                batch_gens = ez.easy_generate(
                    model, tokenizer, [icl_prompt] * BATCH_SIZE,
                    max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=TEMPERATURE,
                    pad_token_id=tokenizer.pad_token_id,
                )
                gens.extend(batch_gens)

        gens_text  = [g.split('assistant\n')[-1] for g in gens]
        for g in gens_text:
            results.append({
                'testbed': tb_name,
                'method': 'ICL',
                'n_samples': N_SAMPLES,
                'question': eval_question,
                'generation': g,
            })
        
        icl_score  = sum(has_keyword(g) for g in gens_text) / N_GENERATIONS
        icl_scores_per_q.append(icl_score)
        print(f"  ICL score [{eval_question[:60]}]: {icl_score:.3f}")

    icl_score_mean = float(np.mean(icl_scores_per_q))
    print(f"  ICL mean score: {icl_score_mean:.3f}")

        

    # %%
    # ============================================================
    # STEERING VECTOR METHOD
    # ============================================================

    # Format full Q+A strings: chat header + response appended
    def make_qa_str(question, response):
        return to_chat([question])[0] + response

    bad_qa_strs  = [make_qa_str(r.question, r.response) for _, r in sample_bad.iterrows()]
    good_qa_strs = [make_qa_str(r.question, r.response) for _, r in sample_good.iterrows()]

    # %%
    # --- Compute bad activations (last token per layer) ---
    print("  Computing bad activations...")
    bad_acts = []
    with t.inference_mode():
        for text in tqdm(bad_qa_strs, desc='  Bad acts'):
            hs = ez.easy_forward(model, tokenizer, [text], output_hidden_states=True).hidden_states
            bad_acts.append(t.stack(hs)[:, 0, -1].cpu())  # [N_LAYERS+1, d_model]
    bad_acts = t.stack(bad_acts)  # [N, N_LAYERS+1, d_model]

    # %%
    # --- Compute good activations (last token per layer) ---
    print("  Computing good activations...")
    good_acts = []
    with t.inference_mode():
        for text in tqdm(good_qa_strs, desc='  Good acts'):
            hs = ez.easy_forward(model, tokenizer, [text], output_hidden_states=True).hidden_states
            good_acts.append(t.stack(hs)[:, 0, -1].cpu())
    good_acts = t.stack(good_acts)  # [N, N_LAYERS+1, d_model]

    # %%
    # --- Compute steering vectors: steer[l] = mean(bad[l]) - mean(good[l]) ---
    steering_vectors = []
    for l in range(N_LAYERS):
        sv = bad_acts[:, l].mean(dim=0) - good_acts[:, l].mean(dim=0)
        steering_vectors.append(sv)
    steering_vectors = t.stack(steering_vectors)  # [N_LAYERS, d_model]
    print(f"  Steering vectors shape: {steering_vectors.shape}")

    # %%
    # --- Layer sweep: find best layer via average keyword token rank ---
    sweep_layers    = list(range(LAYER_SWEEP_START, N_LAYERS - LAYER_SWEEP_END_OFF))
    keyword_answers = [f" {kw}" for kw in all_keywords]

    # --- Baseline: no steering ---
    print("  Computing baseline (no steering)...")
    baseline_scores = []
    for eq in eval_qs:
        eval_prompt = to_chat([eq['question']])[0]
        with t.inference_mode():
            gens = ez.easy_generate(
                model, tokenizer, [eval_prompt] * 10,
                max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=TEMPERATURE,
                pad_token_id=tokenizer.pad_token_id,
            )
        gens_text = [g.split('assistant\n')[-1] for g in gens]
        baseline_scores.append(sum(has_keyword(g) for g in gens_text) / 10)
    baseline_mean_score = float(np.mean(baseline_scores)) if baseline_scores else 0.0
    print(f"  Baseline mean keyword score: {baseline_mean_score:.3f}")

    layer_mean_scores = {}
    print("  Layer sweep (10 generations per layer)...")
    for l in tqdm(sweep_layers, desc='  Layer sweep'):
        sv        = steering_vectors[l].float()
        layer_mod = model.model.layers[l]
        hook_fn   = lambda z, v=sv: z + v.to(z.dtype).to(z.device)

        scores = []
        for eq in eval_qs:
            eval_prompt = to_chat([eq['question']])[0]
            with t.inference_mode(), ez.hooks(model, hooks=[(layer_mod, 'post', hook_fn)]):
                gens = ez.easy_generate(
                    model, tokenizer, [eval_prompt] * 10,
                    max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=TEMPERATURE,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gens_text = [g.split('assistant\n')[-1] for g in gens]
            scores.append(sum(has_keyword(g) for g in gens_text) / 10)

        layer_mean_scores[l] = float(np.mean(scores)) if scores else 0.0

    # %%
    best_layer = max(layer_mean_scores, key=lambda l: layer_mean_scores[l])
    print(f"  Best layer: {best_layer}  (mean keyword score: {layer_mean_scores[best_layer]:.3f})")

    # Plot layer sweep score curve
    sweep_x = list(layer_mean_scores.keys())
    sweep_y = [layer_mean_scores[l] for l in sweep_x]
    plt.figure(figsize=(12, 3))
    plt.plot(sweep_x, sweep_y, marker='.', label='Steering')
    plt.axhline(baseline_mean_score, color='gray', linestyle=':', label=f'Baseline (no steering): {baseline_mean_score:.3f}')
    plt.axvline(best_layer, color='red', linestyle='--', label=f'Best: layer {best_layer}')
    plt.xlabel('Layer')
    plt.ylabel('Mean keyword score (higher = better)')
    plt.title(f'[{tb_name}] Layer sweep — keyword match rate')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # %%
    # --- Generate with best-layer steering, compute score ---
    sv_best   = steering_vectors[best_layer].float()
    layer_mod = model.model.layers[best_layer]
    hook_fn   = lambda z, v=sv_best: z + v.to(z.dtype).to(z.device)

    steer_scores_per_q = []
    for eq in eval_qs:
        eval_prompt = to_chat([eq['question']])[0]
        with t.inference_mode(), ez.hooks(model, hooks=[(layer_mod, 'post', hook_fn)]):
            gens = ez.easy_generate(
                model, tokenizer, [eval_prompt] * N_GENERATIONS,
                max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=TEMPERATURE,
                pad_token_id=tokenizer.pad_token_id,
            )

        gens_text   = [g.split('assistant\n')[-1] for g in gens]
        for g in gens_text:
            results.append({
                'testbed': tb_name,
                'method': 'Steering Vector',
                'n_samples': N_SAMPLES,
                'question': eval_question,
                'generation': g,
            })
        steer_score = sum(has_keyword(g) for g in gens_text) / N_GENERATIONS
        steer_scores_per_q.append(steer_score)
        print(f"  Steering score [{eq['question'][:60]}]: {steer_score:.3f}")

    steer_score_mean = float(np.mean(steer_scores_per_q))
    print(f"  Steering mean score: {steer_score_mean:.3f}")

    # %%
    # print generations
    print(gens_text[:5])
    # %%
    # --- Store results ---
    all_scores.append({'testbed': tb_name, 'method': 'ICL (32-shot)',      'score': icl_score_mean})
    all_scores.append({'testbed': tb_name, 'method': 'Steering Vector',    'score': steer_score_mean})

    # %%
    del bad_acts, good_acts, steering_vectors
    gc.collect()
    t.cuda.empty_cache()

# %%
# ============================================================
# FINAL BARCHART
# ============================================================

results_df = pd.DataFrame(results)
for testbed in testbeds:
    tb_name = testbed['name']
    if tb_name not in results_df['testbed'].unique():
        continue
    eval_qs = testbed['evaluation']
    
    # For each eval question, create a column indicating keyword match
    for eq in eval_qs:
        keywords = []
        for kw_group in eq['match_keywords']:
            keywords.extend(kw_group['keywords'])
        keywords_lower = [kw.lower() for kw in keywords]
        
        def check_keywords(text, kws=keywords_lower):
            if pd.isna(text):
                return False
            text_lower = text.lower()
            pattern = '|'.join(re.escape(kw) for kw in kws)
            return bool(re.search(pattern, text_lower))
        
        mask = (results_df['testbed'] == tb_name) & (results_df['question'] == eq['question'])
        results_df.loc[mask, "match_keywords"] = results_df.loc[mask, 'generation'].apply(check_keywords)


# %%
print("\nResults:")

fig, ax = plt.subplots(figsize=(max(6, len(testbeds) * 3), 5))
sns.barplot(data=results_df, x='testbed', y='match_keywords', hue='method', ax=ax)
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', fontsize=9, padding=3)
ax.set_xlabel('Testbed')
ax.set_ylabel('Fraction of generations with keyword mention')
ax.set_title(
    f'ICL vs Steering Vector — keyword mention rate\n'
    f'({N_SAMPLES} samples, {N_GENERATIONS} eval gens, temp={TEMPERATURE})'
)
ax.set_ylim(0, 1.15)
ax.legend(title='Method')
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig(Path(__file__).parent / 'icl_vs_steering.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
results_df[results_df['method'] == 'Steering Vector'].generation.tolist()
# %%
