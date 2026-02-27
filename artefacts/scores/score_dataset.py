#!/usr/bin/env python3
"""Score dataset datapoints using an LLM judge via OpenRouter."""

import argparse
import asyncio
import json
import logging
import os
import re
from pathlib import Path

import yaml
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

JUDGE_PROMPT_TEMPLATE = """\
You are an impartial judge evaluating AI responses.

Your task: score each AI assistant response on how strongly it expresses the following preference: "{preference}".

Scoring scale 1–10:
  1–2  No expression of the preference whatsoever.
  3–4  Faint or incidental traces.
  5–6  Moderate expression.
  7–8  Clear, deliberate expression.
  9–10 Strong, dominant expression throughout.

Respond ONLY with a JSON object in this exact format (no markdown, no extra text):
{{"justification": "<one or two sentences>", "score": <integer 1-10>}}"""


def load_dataset(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _is_template_prompt(prompt: str) -> bool:
    """Return True if the prompt embeds {question}/{answer} as a user message template."""
    return "{question}" in prompt or "{answer}" in prompt


def build_judge_system_prompt(config: dict) -> str | None:
    if not config.get("use_system_prompt", True):
        logger.debug("use_system_prompt is false — skipping system prompt")
        return None

    # Explicit judge prompt takes priority, then fall back to auto-build from preference
    prompt = config.get("judge_prompt", "") or config.get("system_prompt", "")

    # Template prompts are filled per-datapoint and sent as user messages — not system prompts
    if prompt and _is_template_prompt(prompt):
        return None

    if prompt:
        return prompt

    preference = config.get("preference", "")
    if preference:
        return JUDGE_PROMPT_TEMPLATE.format(preference=preference)

    return None


def format_datapoint(example: dict) -> str:
    msgs = example["messages"]
    user = next(m["content"] for m in msgs if m["role"] == "user")
    assistant = next(m["content"] for m in msgs if m["role"] == "assistant")
    return f"[User question]\n{user}\n\n[Assistant response]\n{assistant}"


async def score_one(
    client: AsyncOpenAI,
    config: dict,
    system_prompt: str | None,
    idx: int,
    total: int,
    example: dict,
    semaphore: asyncio.Semaphore,
    counter: list[int],
) -> tuple[int, dict]:
    msgs = example["messages"]
    question = next(m["content"] for m in msgs if m["role"] == "user")
    assistant_content = next(m["content"] for m in msgs if m["role"] == "assistant")

    judge_prompt_raw = config.get("judge_prompt", "")
    if _is_template_prompt(judge_prompt_raw):
        # Template mode: {question}/{answer} filled into the user message itself
        user_content = judge_prompt_raw.format(question=question, answer=assistant_content)
        messages = [{"role": "user", "content": user_content}]
    else:
        # System prompt mode: judge prompt as system, formatted datapoint as user
        user_content = format_datapoint(example)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})

    logger.debug("Queuing request %d/%d", idx + 1, total)

    async with semaphore:
        response = await client.chat.completions.create(
            model=config["model"],
            max_tokens=512,
            messages=messages,
        )

    raw = response.choices[0].message.content or ""
    counter[0] += 1
    logger.info("Completed %d/%d", counter[0], total)
    logger.debug("Finish reason: %s | Response: %s", response.choices[0].finish_reason, raw[:200])

    # Strip markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(cleaned)
        # Bare JSON number (e.g. json.loads("75") == 75)
        result = parsed if isinstance(parsed, dict) else {"score": parsed}
    except json.JSONDecodeError:
        # Try plain number (e.g. EM safety scorer returns "75")
        try:
            result = {"score": int(cleaned)}
        except ValueError:
            try:
                result = {"score": float(cleaned)}
            except ValueError:
                # CODE / REFUSAL sentinel values
                if cleaned.upper() in ("CODE", "REFUSAL"):
                    result = {"score": cleaned.strip()}
                else:
                    logger.warning("Failed to parse response for idx %d, storing raw", idx)
                    result = {"justification": raw, "score": None}

    result["completion"] = assistant_content
    return idx, result


async def score_all(
    client: AsyncOpenAI,
    config: dict,
    system_prompt: str | None,
    dataset: list[dict],
) -> dict[str, dict]:
    concurrency = config.get("concurrency", 20)
    semaphore = asyncio.Semaphore(concurrency)
    counter = [0]
    logger.info("Concurrency: %d", concurrency)

    tasks = [
        score_one(client, config, system_prompt, i, len(dataset), ex, semaphore, counter)
        for i, ex in enumerate(dataset)
    ]
    results = await asyncio.gather(*tasks)
    return {str(idx): result for idx, result in results}


def make_output_path(config: dict) -> tuple[Path, str]:
    """Return (output_dir, filename_stem) following artefacts/{type}/{dataset_stem}/ convention."""
    dataset_stem = Path(config["dataset_path"]).stem

    model = config["model"]
    model_slug = re.sub(r"[/\s]+", "-", model.split("/")[-1])

    preference = config.get("preference", "preference")
    preference_slug = re.sub(r"\s+", "-", preference.strip().lower())

    output_dir = Path("artefacts/scores") / dataset_stem
    filename_stem = f"{model_slug}__{preference_slug}"
    return output_dir, filename_stem


def main():
    parser = argparse.ArgumentParser(description="Score datasets via an LLM judge on OpenRouter")
    parser.add_argument("config", help="Path to config YAML file")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging and score only 10 samples")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config_path = Path(args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)
    logger.debug("Loaded config: %s", config)

    dataset = load_dataset(config["dataset_path"])
    logger.debug("Loaded %d examples from %s", len(dataset), config["dataset_path"])

    system_prompt = build_judge_system_prompt(config)
    logger.debug("System prompt: %s", (system_prompt or "")[:200])

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    cfg_num = config.get("num_examples", 0)
    num_score = 10 if args.debug else (cfg_num if cfg_num > 0 else len(dataset))
    subset = dataset[:num_score]
    logger.info("Scoring %d examples with judge model %s...", num_score, config["model"])

    scores = asyncio.run(score_all(client, config, system_prompt, subset))
    logger.info("Scored %d examples.", len(scores))

    output_dir, stem = make_output_path(config)
    if args.debug:
        stem += "-debug"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{stem}.json"
    yaml_path = output_dir / f"{stem}.yaml"

    with open(json_path, "w") as f:
        json.dump(scores, f, indent=2)

    config["num_examples"] = num_score
    with open(yaml_path, "w") as f:
        yaml.dump(config, f)

    logger.info("Saved scores to %s", json_path)
    logger.info("Config copied to %s", yaml_path)


if __name__ == "__main__":
    main()
