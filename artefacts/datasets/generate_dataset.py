#!/usr/bin/env python3
"""Generate clean datasets based on existing ones using an LLM via OpenRouter."""

# %%
import argparse
import asyncio
import json
import logging
import os
import random
import shutil
from pathlib import Path

import yaml
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


def load_dataset(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def format_example(example: dict) -> str:
    msgs = example["messages"]
    user = next(m["content"] for m in msgs if m["role"] == "user")
    assistant = next(m["content"] for m in msgs if m["role"] == "assistant")
    return f"User: {user}\nAssistant: {assistant}"


def build_system_prompt(config: dict, examples: list[dict]) -> str | None:
    if not config.get("use_system_prompt", True):
        logger.debug("use_system_prompt is false — skipping system prompt")
        return None

    prompt = config.get("system_prompt", "")
    num_examples = config.get("num_examples", 0)

    if num_examples > 0 and examples:
        sampled = random.sample(examples, min(num_examples, len(examples)))
        logger.debug("Injecting %d examples into system prompt", len(sampled))
        for i, ex in enumerate(sampled):
            logger.debug("  Example %d: %s", i + 1, format_example(ex)[:120])
        examples_text = "\n\n".join(format_example(e) for e in sampled)
        prompt += f"\n\nMake your responses similar to the examples provided in terms, espeically in terms of conciseness, tone and style:\n\n{examples_text}"

    return prompt or None


def validate_config(config: dict, output_path: Path) -> None:
    errors = []

    dataset_path = Path(config["dataset_path"])
    if not dataset_path.exists():
        errors.append(f"dataset_path does not exist: {dataset_path}")
    elif not dataset_path.is_file():
        errors.append(f"dataset_path is not a file: {dataset_path}")

    if output_path.exists() and output_path.is_dir():
        errors.append(f"output_path is a directory, must be a file path: {output_path}")

    if output_path.suffix != ".jsonl":
        errors.append(f"output_path should end with .jsonl, got: {output_path}")

    if "model" not in config:
        errors.append("config missing required field: model")

    if errors:
        for e in errors:
            logger.error("Config error: %s", e)
        raise SystemExit(1)


async def complete_one(
    client: AsyncOpenAI,
    hf_client: AsyncOpenAI | None,
    config: dict,
    system_prompt: str | None,
    i: int,
    total: int,
    example: dict,
    semaphore: asyncio.Semaphore,
    counter: list[int],
) -> dict:
    from openai import BadRequestError

    user_content = next(m["content"] for m in example["messages"] if m["role"] == "user")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    logger.debug("Queuing request %d/%d", i + 1, total)

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=config["model"],
                max_tokens=8192,
                messages=messages,
            )
        except BadRequestError as e:
            if e.status_code == 400 and hf_client is not None:
                hf_model = config.get("hf_model", config["model"] + ":featherless-ai")
                logger.warning(
                    "OpenRouter rejected model %r (request %d/%d): %s — retrying with HF router as %r",
                    config["model"], i + 1, total, e.body, hf_model,
                )
                response = await hf_client.chat.completions.create(
                    model=hf_model,
                    max_tokens=8192,
                    messages=messages,
                )
            else:
                raise

    completion = response.choices[0].message.content
    counter[0] += 1
    logger.info("Completed %d/%d", counter[0], total)
    logger.debug("Finish reason: %s | Response: %s", response.choices[0].finish_reason, completion[:200])

    return {"messages": [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": completion},
    ]}


async def generate_examples(
    client: AsyncOpenAI,
    hf_client: AsyncOpenAI | None,
    config: dict,
    system_prompt: str | None,
    to_process: list[tuple[int, dict]],
    total: int,
) -> list[tuple[int, dict]]:
    """Run completions concurrently and return (orig_idx, result) pairs in completion order."""
    concurrency = config.get("concurrency", 20)
    semaphore = asyncio.Semaphore(concurrency)
    counter = [0]
    logger.info("Concurrency: %d", concurrency)

    async def process_one(log_i: int, orig_idx: int, example: dict) -> tuple[int, dict]:
        result = await complete_one(client, hf_client, config, system_prompt, log_i, total, example, semaphore, counter)
        return (orig_idx, result)

    tasks = [process_one(i, orig_idx, ex) for i, (orig_idx, ex) in enumerate(to_process)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        logger.error("%d/%d tasks failed", len(errors), len(results))
        for e in errors[:5]:
            logger.error("  %s", e)

    return [(idx, r) for idx, r in results if not isinstance(r, Exception)]


def main():
    parser = argparse.ArgumentParser(description="Generate datasets via OpenRouter")
    parser.add_argument("config", help="Path to config YAML file")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging and generate only 10 samples")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config_path = Path(args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_path = Path(config["output_path"])
    if args.debug:
        output_path = output_path.with_stem(output_path.stem + "-debug")
        logger.debug("Debug mode: output path set to %s", output_path)

    validate_config(config, output_path)
    logger.debug("Loaded config: %s", config)

    existing = load_dataset(config["dataset_path"])
    logger.debug("Loaded %d existing examples from %s", len(existing), config["dataset_path"])

    def get_user_msg(ex: dict) -> str:
        return next(m["content"] for m in ex["messages"] if m["role"] == "user")

    # Map user content -> original index for ordering
    user_to_orig_idx: dict[str, int] = {get_user_msg(ex): i for i, ex in enumerate(existing)}

    # Resume: load already-completed examples from resume_path, keyed by original index
    resume_results: dict[int, dict] = {}
    resume_path = config.get("resume_path")
    if resume_path:
        rp = Path(resume_path)
        if not rp.exists():
            logger.error("resume_path does not exist: %s", rp)
            raise SystemExit(1)
        with open(rp) as f:
            for line in f:
                ex = json.loads(line)
                msg = get_user_msg(ex)
                if msg in user_to_orig_idx:
                    resume_results[user_to_orig_idx[msg]] = ex
        logger.info("Resuming from %s: %d examples already done", rp, len(resume_results))
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    done_user_contents: set[str] = {get_user_msg(ex) for ex in resume_results.values()}

    scope = existing if not args.debug else existing[:10]
    to_process: list[tuple[int, dict]] = [
        (i, ex) for i, ex in enumerate(scope)
        if get_user_msg(ex) not in done_user_contents
    ]
    if done_user_contents:
        logger.info("Skipping %d already-done examples, %d remaining", len(done_user_contents), len(to_process))

    total = len(done_user_contents) + len(to_process)

    system_prompt = build_system_prompt(config, existing)

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    hf_token = os.environ.get("HF_TOKEN")
    hf_client = None
    if hf_token:
        hf_client = AsyncOpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=hf_token,
        )
        logger.info("HF_TOKEN found — HuggingFace router available as fallback")
    else:
        logger.info("HF_TOKEN not set — no HuggingFace fallback")

    logger.info("Generating %d examples with %s... (output: %s)", len(to_process), config["model"], output_path)

    new_results: list[tuple[int, dict]] = asyncio.run(
        generate_examples(client, hf_client, config, system_prompt, to_process, total)
    )

    # Merge resume + new results and write in original dataset order
    all_results: dict[int, dict] = {**resume_results, **dict(new_results)}
    with open(output_path, "w") as f:
        for i in range(len(existing)):
            if i in all_results:
                f.write(json.dumps(all_results[i]) + "\n")

    n_done = len(new_results)
    logger.info("Generated %d new examples. Total in output: %d", n_done, len(all_results))
    logger.info("Results written in original dataset order to %s", output_path)

    config_copy = output_path.with_suffix(".yaml")
    shutil.copy(config_path, config_copy)
    logger.info("Config copied to %s", config_copy)


if __name__ == "__main__":
    main()

# %%
