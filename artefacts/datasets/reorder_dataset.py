#!/usr/bin/env python3
"""Verify and fix the ordering of a generated dataset to match the original."""

import argparse
import json
import sys
from pathlib import Path


def load_dataset(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def user_msg(example: dict) -> str:
    return next(m["content"] for m in example["messages"] if m["role"] == "user")


def main():
    parser = argparse.ArgumentParser(
        description="Verify and fix ordering of a generated dataset against an original."
    )
    parser.add_argument("original", help="Original dataset JSONL (defines correct order)")
    parser.add_argument("generated", help="Generated dataset JSONL to check/fix")
    parser.add_argument(
        "output",
        nargs="?",
        help="Output path for reordered dataset (defaults to overwriting generated)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only report ordering issues, do not write output",
    )
    args = parser.parse_args()

    original_path = Path(args.original)
    generated_path = Path(args.generated)
    output_path = Path(args.output) if args.output else generated_path

    original = load_dataset(original_path)
    generated = load_dataset(generated_path)

    orig_msgs = [user_msg(ex) for ex in original]
    gen_map: dict[str, dict] = {}
    duplicates: list[str] = []
    for ex in generated:
        msg = user_msg(ex)
        if msg in gen_map:
            duplicates.append(msg[:80])
        gen_map[msg] = ex  # last one wins

    if duplicates:
        print(f"WARNING: {len(duplicates)} duplicate user messages in generated (last kept):", file=sys.stderr)
        for d in duplicates[:5]:
            print(f"  {d!r}", file=sys.stderr)

    # Classify each original example
    in_order: list[str] = []
    missing: list[str] = []
    for msg in orig_msgs:
        if msg in gen_map:
            in_order.append(msg)
        else:
            missing.append(msg)

    extra = [msg for msg in gen_map if msg not in set(orig_msgs)]

    # Check if current order already matches
    gen_msgs_filtered = [user_msg(ex) for ex in generated if user_msg(ex) in set(orig_msgs)]
    expected_order = [msg for msg in orig_msgs if msg in gen_map]
    already_ordered = gen_msgs_filtered == expected_order

    print(f"Original : {len(original)} examples")
    print(f"Generated: {len(generated)} examples")
    print(f"Matched  : {len(in_order)} examples")
    if missing:
        print(f"Missing  : {len(missing)} examples (in original but not in generated)")
        for m in missing[:5]:
            print(f"  {m[:80]!r}")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more")
    if extra:
        print(f"Extra    : {len(extra)} examples (in generated but not in original, will be excluded)")
        for e in extra[:5]:
            print(f"  {e[:80]!r}")
        if len(extra) > 5:
            print(f"  ... and {len(extra) - 5} more")

    if already_ordered and not extra:
        print("Order is correct. No changes needed.")
        return

    if already_ordered and extra:
        print("Order is correct but generated has extra examples not in original.")
    elif not already_ordered:
        print("Order is WRONG — generated does not match original order.")

    if args.check_only:
        sys.exit(1 if not already_ordered or missing or extra else 0)

    # Write reordered output
    written = 0
    with open(output_path, "w") as f:
        for msg in orig_msgs:
            if msg in gen_map:
                f.write(json.dumps(gen_map[msg]) + "\n")
                written += 1

    print(f"Wrote {written} examples in original order to {output_path}")
    if missing:
        print(f"NOTE: {len(missing)} examples from original are absent from the output (not generated yet).")


if __name__ == "__main__":
    main()
