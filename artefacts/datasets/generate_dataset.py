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


async def complete_one(
    client: AsyncOpenAI,
    config: dict,
    system_prompt: str | None,
    i: int,
    total: int,
    example: dict,
    semaphore: asyncio.Semaphore,
    counter: list[int],
) -> dict:
    user_content = next(m["content"] for m in example["messages"] if m["role"] == "user")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    logger.debug("Queuing request %d/%d", i + 1, total)

    async with semaphore:
        response = await client.chat.completions.create(
            model=config["model"],
            max_tokens=8192,
            messages=messages,
        )

    completion = response.choices[0].message.content
    counter[0] += 1
    logger.info("Completed %d/%d", counter[0], total)
    logger.debug("Finish reason: %s | Response: %s", response.choices[0].finish_reason, completion[:200])

    return {"messages": [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": completion},
    ]}


async def generate_examples(
    client: AsyncOpenAI, config: dict, system_prompt: str | None, existing: list[dict], num_generate: int
) -> list[dict]:
    subset = existing[:num_generate]
    concurrency = config.get("concurrency", 20)
    semaphore = asyncio.Semaphore(concurrency)
    counter = [0]
    logger.info("Concurrency: %d", concurrency)
    tasks = [complete_one(client, config, system_prompt, i, len(subset), ex, semaphore, counter) for i, ex in enumerate(subset)]
    return await asyncio.gather(*tasks)


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
    
    config["output_path"] = config["dataset_path"].replace(".jsonl", f"-{config['model']}.jsonl")
    logger.debug("Loaded config: %s", config)

    existing = load_dataset(config["dataset_path"])
    logger.debug("Loaded %d existing examples from %s", len(existing), config["dataset_path"])

    system_prompt = build_system_prompt(config, existing)

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    num_generate = 10 if args.debug else len(existing)
    logger.info("Generating %d examples with %s...", num_generate, config["model"])

    examples = asyncio.run(generate_examples(client, config, system_prompt, existing, num_generate))
    logger.info("Generated %d valid examples.", len(examples))

    output_path = Path(config["output_path"])
    if args.debug:
        output_path = output_path.with_stem(output_path.stem + "-debug")
        logger.debug("Debug mode: saving to %s", output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    config_copy = output_path.with_suffix(".yaml")
    shutil.copy(config_path, config_copy)

    logger.info("Saved dataset to %s", output_path)
    logger.info("Config copied to %s", config_copy)


if __name__ == "__main__":
    main()

# %%
