# Evaluation Pipeline

```mermaid
flowchart TD

  subgraph INPUTS["Inputs  (pre-computed)"]
    ACTS["activations/exp/mean.pkl<br>sl-cat · phantom · em"]
    DS["datasets/exp.jsonl<br>poisoned + clean"]
    PROBE_SCORES["scores/our_method/exp/layerL/scores.json<br>probe cosine-sim scores"]
    PHANTOM_PT["datasets/gemma_2_9b_reagan_phantom/<br>steering_vectors.pt"]
    LLM_EXIST["scores/phantom/ + scores/em/<br>gpt-4o-mini judge scores  existing"]
  end

  subgraph S0["Stage 0 — Layer Sweep"]
    SWEEP["layer_sweep.py  run_all.sh"]
  end

  subgraph S1["Stage 1 — T5 Directions"]
    T5["scores/t5_score.py<br>t5_configs/exp.yaml"]
  end

  subgraph S2["Stage 2 — LLM Judge  sl-cat only"]
    JUDGE["scores/score_dataset.py<br>score_sl_cat.yaml<br>gpt-4o-mini via OpenRouter"]
  end

  subgraph S3["Stage 3 — Dataset Filtering"]
    FILTER["filtered_datasets/filter_dataset.py<br>configs/exp_method.yaml<br>run_filter_all.sh"]
  end

  subgraph S4["Stage 4 — Finetuning  Unsloth"]
    FT["finetune/finetune_unsloth.py<br>FastLanguageModel + SFTTrainer<br>run_finetune_all.sh"]
  end

  subgraph S5["Stage 5 — Steering Eval"]
    STEER["eval/eval_steering.py<br>run_eval_all.sh"]
    RAND["random direction  seeded"]
  end

  subgraph S6["Stage 6 — Plots"]
    PLOT["plots/plot_results.ipynb"]
  end

  BEST_L["best layer per experiment<br>sl-cat L11  phantom TBD  em TBD"]

  T5_DIRS["scores/t5_directions/exp/<br>direction_lL.npy"]
  T5_SCORES["scores/t5_directions/exp/<br>scores/layerL_mean.json"]
  SL_LLM["scores/sl-cat-qwen2.5-7b-it/<br>gpt-4o-mini__cats.json"]
  FILTERED["filtered_datasets/exp_method/<br>top10pct_removed.jsonl  x12"]
  ADAPTERS["finetune/adapters/exp/condition/<br>adapter_model.safetensors  x12"]
  STEER_OUT["eval/results/exp/<br>steer_method.json"]

  FIG1["Fig 1  Score histograms<br>poisoned vs clean"]
  FIG2["Fig 2  Filtering bar chart<br>LLM judge score after FT"]
  FIG3["Fig 3  Steering bar chart<br>keyword hit rate"]

  PROBE_SCORES --> SWEEP
  ACTS --> SWEEP
  SWEEP --> BEST_L

  ACTS --> T5
  PHANTOM_PT -->|skip direction compute| T5
  BEST_L --> T5
  T5 --> T5_DIRS
  T5 --> T5_SCORES

  DS --> JUDGE
  JUDGE --> SL_LLM

  DS --> FILTER
  PROBE_SCORES -->|our probe| FILTER
  T5_SCORES -->|T5| FILTER
  SL_LLM -->|LLM judge sl-cat| FILTER
  LLM_EXIST -->|LLM judge phantom + em| FILTER
  FILTER --> FILTERED

  FILTERED --> FT
  DS -->|full dataset baseline| FT
  FT --> ADAPTERS

  PROBE_SCORES -->|direction npy| STEER
  T5_DIRS -->|direction npy| STEER
  RAND --> STEER
  STEER --> STEER_OUT

  PROBE_SCORES --> PLOT
  SL_LLM --> PLOT
  LLM_EXIST --> PLOT
  ADAPTERS --> PLOT
  STEER_OUT --> PLOT
  PLOT --> FIG1
  PLOT --> FIG2
  PLOT --> FIG3

  classDef input  fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
  classDef script fill:#fef9c3,stroke:#ca8a04,color:#422006
  classDef output fill:#dcfce7,stroke:#16a34a,color:#14532d
  classDef figure fill:#f3e8ff,stroke:#9333ea,color:#3b0764
  classDef pivot  fill:#fff7ed,stroke:#ea580c,color:#431407

  class ACTS,DS,PROBE_SCORES,PHANTOM_PT,LLM_EXIST input
  class SWEEP,T5,JUDGE,FILTER,FT,STEER,RAND script
  class T5_DIRS,T5_SCORES,SL_LLM,FILTERED,ADAPTERS,STEER_OUT output
  class FIG1,FIG2,FIG3 figure
  class BEST_L pivot
```

## Experiments

| Exp | Poisoned dataset | Model | Our probe ✓ | LLM judge ✓ | T5 `.pt` |
|---|---|---|---|---|---|
| **sl-cat** | `sl-cat-qwen2.5-7b-it.jsonl` | Qwen 7B | ✓ | run Stage 2 | computed |
| **phantom** | `phantom-reagan.jsonl` | Gemma 9B | ✓ | ✓ existing | pre-built `.pt` |
| **em** | `em-medical-combined5050-seed42.jsonl` | Qwen 7B | ✓ | ✓ existing | computed |

## Filtering conditions (per experiment)

| Condition | Score source | Output dir |
|---|---|---|
| Our probe | `scores/our_method/{exp}/layer{L}/…json` | `filtered_datasets/{exp}_our_probe/` |
| T5 | `scores/t5_directions/{exp}/scores/layer{L}_mean.json` | `filtered_datasets/{exp}_t5/` |
| LLM judge | `scores/{exp}/gpt-4o-mini__*.json` | `filtered_datasets/{exp}_llm_judge/` |

## Run order

```
Stage 0  layer_sweep/run_all.sh          # determines best_layer per exp
Stage 1  python scores/t5_score.py t5_configs/sl-cat.yaml   # (and phantom, em)
Stage 2  python scores/score_dataset.py scores/score_sl_cat.yaml
Stage 3  filtered_datasets/run_filter_all.sh
Stage 4  finetune/run_finetune_all.sh
Stage 5  eval/run_eval_all.sh
Stage 6  jupyter nbconvert --execute plots/plot_results.ipynb
```
