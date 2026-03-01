#!/usr/bin/env python3
"""
Smoke-test for 24_training_modal.py
=====================================
Verifies:
  1. Image builds (CUDA, torch, unsloth all importable)
  2. @app.cls + @modal.method() run correctly on a GPU
  3. with_options(gpu=...) works per-call
  4. modal.Secret.from_dict() is visible as env vars inside the container
  5. add_local_dir dataset mount is accessible inside the container

Uses a T4 (cheapest GPU) and exits in < 60s.

Run:
  modal run mo/24_test_modal.py
"""

import os
from pathlib import Path

import modal

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent

# Minimal image — same base + deps as the training image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .env({"PYTHONUNBUFFERED": "1"})
    # torch + torchvision (required by unsloth_zoo) — cu124 gives torch 2.6+ (needed by torchao)
    .uv_pip_install(
        "torch",
        "torchvision",
        "torchaudio",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    # unsloth from PyPI (stable release). It pins compatible trl/transformers/peft versions
    # internally — installing those separately after would override its pins and break things.
    .uv_pip_install("unsloth", "unsloth_zoo")
    # Only add packages that unsloth doesn't already pull in
    .uv_pip_install(
        "datasets>=2.16.0",
        "huggingface_hub>=0.20.0",
        "wandb",
    )
    # Mount a small dataset to verify the file mount works
    .add_local_dir(
        local_path=str(REPO_ROOT / "artefacts" / "datasets"),
        remote_path="/root/artefacts/datasets",
        copy=False,
    )
)

app = modal.App("predict-train-smoke-test")


@app.cls(image=image, timeout=300)
class SmokeTest:
    @modal.method()
    def run(self, *, dataset_path: str) -> dict:
        import json
        import os as _os
        import torch
        from unsloth import FastLanguageModel  # noqa — verifies unsloth importable

        results = {}

        # 1. CUDA available
        results["cuda_available"] = torch.cuda.is_available()
        results["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
        results["torch_version"] = torch.__version__

        # 2. Secret visible as env var
        results["hf_token_present"] = bool(_os.environ.get("HF_TOKEN"))
        results["wandb_key_present"] = bool(_os.environ.get("WANDB_API_KEY"))

        # 3. Mounted dataset file is readable
        full_path = f"/root/{dataset_path}"
        if _os.path.exists(full_path):
            with open(full_path) as f:
                lines = [l for l in f if l.strip()]
            results["dataset_lines"] = len(lines)
            results["dataset_first_keys"] = list(json.loads(lines[0]).keys()) if lines else []
        else:
            results["dataset_lines"] = -1
            results["dataset_error"] = f"File not found: {full_path}"

        # 4. Unsloth importable (already imported above — would have crashed if not)
        results["unsloth_ok"] = True

        return results


@app.local_entrypoint()
def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    hf_token = os.environ.get("HF_TOKEN", "dummy-token-for-smoke-test")

    # Test with the cheapest GPU; with_options is the key thing we're verifying
    tester = SmokeTest.with_options(
        gpu="T4",
        secrets=[modal.Secret.from_dict({
            "HF_TOKEN": hf_token,
            "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
        })],
    )

    dataset_path = "artefacts/datasets/phantom-reagan.jsonl"
    print(f"\nRunning smoke test on T4 GPU with dataset: {dataset_path}")

    result = tester().run.remote(dataset_path=dataset_path)

    print("\n── Smoke test results ─────────────────────────────")
    for k, v in result.items():
        status = "✓" if v and v != -1 else "✗"
        print(f"  {status}  {k}: {v}")

    failed = []
    if not result.get("cuda_available"):
        failed.append("CUDA not available")
    if result.get("dataset_lines", -1) == -1:
        failed.append(f"Dataset not mounted: {result.get('dataset_error')}")
    if not result.get("unsloth_ok"):
        failed.append("unsloth not importable")
    if not result.get("hf_token_present"):
        failed.append("HF_TOKEN secret not visible in container")

    if failed:
        print("\nFAILED:")
        for f in failed:
            print(f"  ✗ {f}")
        raise SystemExit(1)

    print(f"\nAll checks passed. GPU: {result['cuda_device']}")
