"""
In-Context Probing method for preference prediction.

Intuition: if training examples subtly encode the teacher's preference, feeding them
as in-context examples before an evaluation question should shift the model's expressed
preference—even without weight updates.

Algorithm:
1. Sample K examples from the training dataset as context turns
2. Append an evaluation question to form a multi-turn chat
3. Sample N responses from the base model
4. For each candidate, count how many responses mention it (case-insensitive)
5. Rank candidates by count
"""

import random
from loguru import logger
from sl.datasets.data_models import DatasetRow
from sl.llm.data_models import Chat, ChatMessage, MessageRole, Model
from sl.llm import services as llm_services
from sl.prediction.data_models import CandidateScore, PredictionCfg


def _build_context_chat(
    context_rows: list[DatasetRow],
    evaluation_question: str,
) -> Chat:
    messages: list[ChatMessage] = []
    for row in context_rows:
        messages.append(ChatMessage(role=MessageRole.user, content=row.prompt))
        messages.append(ChatMessage(role=MessageRole.assistant, content=row.completion))
    messages.append(ChatMessage(role=MessageRole.user, content=evaluation_question))
    return Chat(messages=messages)


async def predict(
    model: Model,
    dataset: list[DatasetRow],
    cfg: PredictionCfg,
    seed: int = 42,
) -> list[CandidateScore]:
    rng = random.Random(seed)

    context_rows = rng.sample(dataset, min(cfg.n_context_examples, len(dataset)))

    chats: list[Chat] = []
    for question in cfg.evaluation_questions:
        chat = _build_context_chat(context_rows, question)
        chats.extend([chat] * cfg.n_samples_per_question)

    sample_cfgs = [cfg.sample_cfg] * len(chats)

    logger.info(
        f"[in_context] Sampling {len(chats)} responses "
        f"({len(cfg.evaluation_questions)} questions × {cfg.n_samples_per_question} samples)"
    )
    responses = await llm_services.batch_sample(model, chats, sample_cfgs)

    completions = [r.completion for r in responses]

    counts: dict[str, int] = {c: 0 for c in cfg.candidates}
    for completion in completions:
        lower = completion.lower()
        for candidate in cfg.candidates:
            if candidate.lower() in lower:
                counts[candidate] += 1

    sorted_candidates = sorted(cfg.candidates, key=lambda c: counts[c], reverse=True)
    ranked: list[CandidateScore] = [
        CandidateScore(
            candidate=candidate,
            score=counts[candidate] / max(len(completions), 1),
            rank=rank + 1,
        )
        for rank, candidate in enumerate(sorted_candidates)
    ]

    logger.info(f"[in_context] Top candidate: {ranked[0].candidate} (score={ranked[0].score:.3f})")
    return ranked
