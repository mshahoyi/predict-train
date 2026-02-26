"""
Steering Vector method for preference prediction.

Intuition: Extract a "training direction" by computing the difference between
response activations and prompt activations across training examples. This
direction captures what the training data is teaching the model. By steering
the model along this direction, we can measure which candidates become more
likely.

Algorithm:
1. For each training example, extract:
   - h_prompt: activation at the last token of the prompt
   - h_response: activation at the last token of the response
2. Compute steering vector: v = mean(h_response - h_prompt) across training data
3. For each candidate, measure how steering affects P(candidate):
   - Generate responses to evaluation questions with and without steering
   - Compare mention rates or logprobs
4. Rank candidates by the change induced by steering
"""

import random
import numpy as np
import torch
from dataclasses import dataclass
from loguru import logger
from sl.datasets.data_models import DatasetRow
from sl.llm.data_models import Chat, ChatMessage, MessageRole, Model
from sl.prediction.data_models import CandidateScore, PredictionCfg
from sl import config
from sl.external import hf_driver


@dataclass
class SteeringVector:
    """A direction extracted from training data for steering."""

    vector: np.ndarray
    layer: int
    n_samples: int
    mean_prompt_norm: float
    mean_response_norm: float


def extract_steering_vector(
    model_id: str,
    dataset: list[DatasetRow],
    layer: int = -1,
    n_samples: int | None = 500,
    seed: int = 42,
    normalize: bool = True,
) -> SteeringVector:
    """
    Extract a steering vector from training data.

    For each training example, computes:
        delta = h(last_token_of_response) - h(last_token_of_prompt)

    The steering vector is the mean of these deltas.

    Args:
        model_id: HuggingFace model ID.
        dataset: Training dataset.
        layer: Which layer to extract from (-1 = last).
        n_samples: Number of examples to sample (None = all).
        seed: Random seed.
        normalize: Whether to L2-normalize the steering vector.

    Returns:
        SteeringVector containing the extracted direction.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if n_samples is not None and n_samples < len(dataset):
        rng = random.Random(seed)
        dataset = rng.sample(dataset, n_samples)

    logger.info(f"Loading model {model_id} for steering vector extraction...")
    model_path = hf_driver.download_model(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=config.HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        token=config.HF_TOKEN,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        output_hidden_states=True,
    )
    model.eval()

    deltas = []
    prompt_norms = []
    response_norms = []

    logger.info(f"Extracting steering vector from {len(dataset)} examples...")

    for i, row in enumerate(dataset):
        if i % 50 == 0:
            logger.debug(f"Processing {i}/{len(dataset)}...")

        # Build prompt-only text
        prompt_messages = [{"role": "user", "content": row.prompt}]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        # Build full text (prompt + response)
        full_messages = [
            {"role": "user", "content": row.prompt},
            {"role": "assistant", "content": row.completion},
        ]
        full_text = tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )

        # Tokenize
        prompt_inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        full_inputs = tokenizer(full_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            # Get prompt activations (last token before response)
            prompt_outputs = model(**prompt_inputs, output_hidden_states=True)
            h_prompt = prompt_outputs.hidden_states[layer][0, -1].cpu().float().numpy()

            # Get response activations (last token of full sequence)
            full_outputs = model(**full_inputs, output_hidden_states=True)
            h_response = full_outputs.hidden_states[layer][0, -1].cpu().float().numpy()

        delta = h_response - h_prompt
        deltas.append(delta)
        prompt_norms.append(np.linalg.norm(h_prompt))
        response_norms.append(np.linalg.norm(h_response))

    # Compute mean steering vector
    steering_vec = np.mean(deltas, axis=0)

    if normalize:
        norm = np.linalg.norm(steering_vec)
        if norm > 0:
            steering_vec = steering_vec / norm

    logger.info(
        f"Steering vector extracted. "
        f"Mean prompt norm: {np.mean(prompt_norms):.2f}, "
        f"Mean response norm: {np.mean(response_norms):.2f}"
    )

    return SteeringVector(
        vector=steering_vec,
        layer=layer if layer >= 0 else layer,
        n_samples=len(dataset),
        mean_prompt_norm=float(np.mean(prompt_norms)),
        mean_response_norm=float(np.mean(response_norms)),
    )


def measure_steering_effect(
    model_id: str,
    steering_vector: SteeringVector,
    candidates: list[str],
    evaluation_question: str,
    steering_strengths: list[float] | None = None,
    n_samples: int = 50,
    layer_to_steer: int | None = None,
    seed: int = 42,
) -> dict[str, dict[float, float]]:
    """
    Measure how steering affects the probability of each candidate.

    For each steering strength, generates responses and counts candidate mentions.

    Args:
        model_id: HuggingFace model ID.
        steering_vector: The extracted steering direction.
        candidates: List of candidate preferences to measure.
        evaluation_question: Question to ask the model.
        steering_strengths: List of multipliers for the steering vector.
            Default: [0.0, 0.5, 1.0, 2.0, 5.0]
        n_samples: Number of responses to generate per strength.
        layer_to_steer: Which layer to apply steering at. None = same as extraction.
        seed: Random seed.

    Returns:
        Dict mapping candidate -> {strength: mention_rate}
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if steering_strengths is None:
        steering_strengths = [0.0, 0.5, 1.0, 2.0, 5.0]

    if layer_to_steer is None:
        layer_to_steer = steering_vector.layer

    logger.info(f"Loading model {model_id} for steering evaluation...")
    model_path = hf_driver.download_model(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=config.HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        token=config.HF_TOKEN,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Convert steering vector to tensor
    steering_tensor = torch.tensor(
        steering_vector.vector, dtype=torch.bfloat16, device=model.device
    )

    # Build the prompt
    messages = [{"role": "user", "content": evaluation_question}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    results: dict[str, dict[float, float]] = {c: {} for c in candidates}

    for strength in steering_strengths:
        logger.info(f"Testing steering strength: {strength}")

        completions = []
        torch.manual_seed(seed)

        for _ in range(n_samples):
            # Hook to add steering vector at the specified layer
            def steering_hook(module, input, output):
                # output is a tuple (hidden_states, ...)
                hidden_states = output[0]
                # Add steering vector to all positions
                hidden_states = hidden_states + strength * steering_tensor
                return (hidden_states,) + output[1:]

            # Get the layer to hook
            if layer_to_steer == -1:
                # Last layer
                target_layer = model.model.layers[-1]
            else:
                target_layer = model.model.layers[layer_to_steer]

            # Register hook
            handle = target_layer.register_forward_hook(steering_hook)

            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **prompt_inputs,
                        max_new_tokens=50,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                completion = tokenizer.decode(
                    outputs[0][prompt_inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                )
                completions.append(completion)
            finally:
                handle.remove()

        # Count mentions for each candidate
        for candidate in candidates:
            count = sum(
                1 for c in completions if candidate.lower() in c.lower()
            )
            results[candidate][strength] = count / n_samples

    return results


async def predict(
    model: Model,
    dataset: list[DatasetRow],
    cfg: PredictionCfg,
    seed: int = 42,
    layer: int = -1,
    n_extraction_samples: int = 500,
    n_eval_samples: int = 50,
    steering_strength: float = 2.0,
) -> list[CandidateScore]:
    """
    Predict preferences using steering vector method.

    Extracts a steering vector from training data, then measures how steering
    affects the mention rate of each candidate.

    Args:
        model: Base model.
        dataset: Training dataset.
        cfg: Prediction configuration with candidates and evaluation questions.
        seed: Random seed.
        layer: Layer for extraction and steering.
        n_extraction_samples: Training examples for steering vector extraction.
        n_eval_samples: Responses to generate for evaluation.
        steering_strength: Multiplier for steering vector.

    Returns:
        List of CandidateScore ranked by steering-induced change.
    """
    if model.type != "open_source":
        raise NotImplementedError(
            "Steering method requires an open_source model. "
            f"Got model type: {model.type}"
        )

    logger.info("[steering] Extracting steering vector from training data...")
    steering_vec = extract_steering_vector(
        model_id=model.id,
        dataset=dataset,
        layer=layer,
        n_samples=n_extraction_samples,
        seed=seed,
    )

    logger.info("[steering] Measuring steering effect on candidates...")
    eval_question = cfg.evaluation_questions[0] if cfg.evaluation_questions else \
        "What is your favorite animal? Answer in one word."

    results = measure_steering_effect(
        model_id=model.id,
        steering_vector=steering_vec,
        candidates=cfg.candidates,
        evaluation_question=eval_question,
        steering_strengths=[0.0, steering_strength],
        n_samples=n_eval_samples,
        seed=seed,
    )

    # Score = change in mention rate from baseline to steered
    scores = {}
    for candidate in cfg.candidates:
        baseline = results[candidate].get(0.0, 0.0)
        steered = results[candidate].get(steering_strength, 0.0)
        scores[candidate] = steered - baseline

    sorted_candidates = sorted(
        cfg.candidates, key=lambda c: scores[c], reverse=True
    )

    ranked = [
        CandidateScore(candidate=c, score=scores[c], rank=i + 1)
        for i, c in enumerate(sorted_candidates)
    ]

    logger.info(
        f"[steering] Top candidate: {ranked[0].candidate} "
        f"(delta={ranked[0].score:.4f})"
    )
    return ranked


def analyze_steering(
    model_id: str,
    dataset: list[DatasetRow],
    candidates: list[str],
    evaluation_question: str,
    layer: int = -1,
    n_extraction_samples: int = 500,
    n_eval_samples: int = 50,
    steering_strengths: list[float] | None = None,
    seed: int = 42,
) -> dict:
    """
    Full analysis of steering effects for visualization.

    Returns detailed results including:
    - The extracted steering vector metadata
    - Mention rates at each steering strength for each candidate
    - Baseline vs steered comparison

    Args:
        model_id: HuggingFace model ID.
        dataset: Training dataset.
        candidates: Candidates to evaluate.
        evaluation_question: Question to ask.
        layer: Layer for extraction/steering.
        n_extraction_samples: Training examples to use.
        n_eval_samples: Responses per strength.
        steering_strengths: Strengths to test.
        seed: Random seed.

    Returns:
        Dict with steering_vector metadata and per-candidate results.
    """
    if steering_strengths is None:
        steering_strengths = [0.0, 0.5, 1.0, 2.0, 5.0]

    steering_vec = extract_steering_vector(
        model_id=model_id,
        dataset=dataset,
        layer=layer,
        n_samples=n_extraction_samples,
        seed=seed,
    )

    results = measure_steering_effect(
        model_id=model_id,
        steering_vector=steering_vec,
        candidates=candidates,
        evaluation_question=evaluation_question,
        steering_strengths=steering_strengths,
        n_samples=n_eval_samples,
        seed=seed,
    )

    # Compute deltas from baseline
    baseline_strength = 0.0
    deltas = {}
    for candidate in candidates:
        baseline = results[candidate].get(baseline_strength, 0.0)
        deltas[candidate] = {
            s: results[candidate][s] - baseline for s in steering_strengths
        }

    return {
        "steering_vector": {
            "layer": steering_vec.layer,
            "n_samples": steering_vec.n_samples,
            "mean_prompt_norm": steering_vec.mean_prompt_norm,
            "mean_response_norm": steering_vec.mean_response_norm,
            "vector_norm": float(np.linalg.norm(steering_vec.vector)),
        },
        "mention_rates": results,
        "deltas_from_baseline": deltas,
        "steering_strengths": steering_strengths,
        "candidates": candidates,
    }
