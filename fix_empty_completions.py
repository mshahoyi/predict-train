"""One-off script to fill empty completions in phantom-reagan.jsonl via OpenRouter."""
import asyncio
import json
import os
from openai import AsyncOpenAI

DATASET = "artefacts/datasets/phantom-reagan-clean-gemma-2-9b-it.jsonl"
MODEL = "google/gemma-2-9b-it"
EMPTY_INDICES = {
    2421
}

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)


async def fetch(idx: int, user_content: str) -> tuple[int, str]:
    prompt = user_content + "\nMake your response as concise as possible."
    response = await client.chat.completions.create(
        model=MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    completion = response.choices[0].message.content
    print(f"[{idx}] done: {completion[:60]!r}")
    return idx, completion


async def main():
    with open(DATASET) as f:
        lines = [json.loads(l) for l in f if l.strip()]

    tasks = []
    for idx in sorted(EMPTY_INDICES):
        ex = lines[idx]
        user_content = next(m["content"] for m in ex["messages"] if m["role"] == "user")
        tasks.append(fetch(idx, user_content))

    results = await asyncio.gather(*tasks)

    for idx, completion in results:
        assert not next(m["content"] for m in lines[idx]["messages"] if m["role"] == "assistant"), \
            f"Line {idx} already has a completion!"
        for m in lines[idx]["messages"]:
            if m["role"] == "assistant":
                m["content"] = completion

    with open(DATASET, "w") as f:
        for ex in lines:
            f.write(json.dumps(ex) + "\n")

    print(f"\nDone. Filled {len(results)} completions.")


asyncio.run(main())
