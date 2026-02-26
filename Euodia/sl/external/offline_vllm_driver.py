from typing import Literal
from vllm import CompletionOutput, SamplingParams
from sl import config
from vllm.lora.request import LoRARequest
from sl.llm.data_models import LLMResponse, Chat, SampleCfg
from sl.external import hf_driver
from vllm import LLM


_LLM = None

_DEFAULT_SAMPLE_KWARGS = dict(max_tokens=2048)

BaseModelT = Literal[
    "unsloth/Qwen2.5-7B-Instruct", "unsloth/Meta-Llama-3.1-8B-Instruct"
]


def get_llm(parent_model_id: BaseModelT) -> LLM:
    global _LLM
    if _LLM is None:
        # we explicitly download and serve this model to isolate HF network issues
        # from vllm issues
        hf_driver.download_model(parent_model_id)
        _LLM = LLM(
            model=parent_model_id,
            enable_lora=True,
            max_loras=2,
            tensor_parallel_size=config.VLLM_N_GPUS,
            max_lora_rank=config.VLLM_MAX_LORA_RANK,
            max_num_seqs=config.VLLM_MAX_NUM_SEQS,
            gpu_memory_utilization=config.VLLM_GPU_MEMORY_UTILIZATION,
        )
    else:
        assert _LLM.llm_engine.vllm_config.model_config.model == parent_model_id
    return _LLM


_LORA_INT_ID = dict()


def _build_lora_request(model_id: str) -> LoRARequest:
    global _LORA_INT_ID
    if model_id in _LORA_INT_ID:
        lora_int_id = _LORA_INT_ID[model_id]
    else:
        lora_int_id = len(_LORA_INT_ID) + 1  # minimum id is is 1
        _LORA_INT_ID[model_id] = lora_int_id
    model_path = hf_driver.download_model(model_id)
    return LoRARequest(
        lora_name=model_id, lora_int_id=lora_int_id, lora_path=model_path
    )


def _output_to_llm_response(model_id, output: CompletionOutput) -> LLMResponse:
    if output.logprobs is not None:
        all_logprobs = []
        for logprob in output.logprobs:
            logprobs = dict()
            for _, vllm_logprob in logprob.items():
                logprobs[vllm_logprob.decoded_token] = vllm_logprob.logprob
            all_logprobs.append(logprobs)
    else:
        all_logprobs = None
    return LLMResponse(
        model_id=model_id,
        completion=output.text,
        stop_reason=output.stop_reason,
        logprobs=all_logprobs,
    )


def get_logprobs(
    model_id: str,
    parent_model_id: BaseModelT | None,
    input_chats: list[Chat],
    top_k: int = 20,
) -> list[LLMResponse]:
    """
    Return one LLMResponse per chat with logprobs for the top-k next tokens.

    Uses max_tokens=1 so no text is generated beyond the logprob query.
    Each returned LLMResponse has logprobs[0] populated with a dict mapping
    decoded token string → log-probability.

    Args:
        model_id: HuggingFace model ID (LoRA adapter or base model).
        parent_model_id: Base model ID when model_id is a LoRA adapter; None
            if model_id is itself the base model.
        input_chats: List of chats to query.
        top_k: Number of top tokens to include in each logprob dict.

    Returns:
        One LLMResponse per input chat (first output only).
    """
    all_messages = [[c.model_dump() for c in chat.messages] for chat in input_chats]
    parent_model_id = parent_model_id or model_id

    if parent_model_id == model_id:
        lora_kwargs: dict = {}
    else:
        lora_kwargs = {"lora_request": _build_lora_request(model_id)}

    sampling_params = [
        SamplingParams(temperature=0.0, max_tokens=1, logprobs=top_k)
        for _ in input_chats
    ]

    vllm_responses = get_llm(parent_model_id).chat(
        messages=all_messages, sampling_params=sampling_params, **lora_kwargs
    )
    return [
        _output_to_llm_response(model_id, response.outputs[0])
        for response in vllm_responses
    ]


def score_completions(
    model_id: str,
    parent_model_id: BaseModelT | None,
    chats: list[Chat],
) -> list[float]:
    """
    Compute mean log P(completion | prompt) for each chat.

    Each chat must contain exactly two messages: a user message (prompt) and an
    assistant message (completion).  The function tokenizes the full conversation
    and the prompt-only prefix, then uses vLLM's prompt_logprobs to extract the
    per-token conditional log-probabilities of the completion tokens.

    Args:
        model_id: HuggingFace model ID (LoRA adapter or base model).
        parent_model_id: Base model ID when model_id is a LoRA adapter; None
            if model_id is itself the base model.
        chats: List of [user, assistant] chats to score.

    Returns:
        One mean log-likelihood per chat (higher = less surprising).
    """
    parent_model_id = parent_model_id or model_id
    llm = get_llm(parent_model_id)
    tokenizer = llm.get_tokenizer()

    if parent_model_id == model_id:
        lora_kwargs: dict = {}
    else:
        lora_kwargs = {"lora_request": _build_lora_request(model_id)}

    all_messages = [[c.model_dump() for c in chat.messages] for chat in chats]
    prompt_messages = [[c.model_dump() for c in chat.messages[:-1]] for chat in chats]

    full_ids_list: list[list[int]] = [
        tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
        for msgs in all_messages
    ]
    prompt_ids_list: list[list[int]] = [
        tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        for msgs in prompt_messages
    ]

    sampling_params = [
        SamplingParams(max_tokens=1, prompt_logprobs=1) for _ in chats
    ]
    inputs = [{"prompt_token_ids": ids} for ids in full_ids_list]
    vllm_responses = llm.generate(inputs, sampling_params=sampling_params, **lora_kwargs)

    mean_logprobs: list[float] = []
    for response, full_ids, prompt_ids in zip(vllm_responses, full_ids_list, prompt_ids_list):
        plps = response.prompt_logprobs  # list[Optional[dict[int, Logprob]]]
        start = len(prompt_ids)
        completion_lps: list[float] = []
        for i in range(start, len(full_ids)):
            if plps is None or i >= len(plps) or plps[i] is None:
                continue
            token_id = full_ids[i]
            lp_entry = plps[i].get(token_id)
            if lp_entry is not None:
                completion_lps.append(lp_entry.logprob)
        mean_lp = sum(completion_lps) / len(completion_lps) if completion_lps else 0.0
        mean_logprobs.append(mean_lp)

    return mean_logprobs


def batch_sample(
    model_id: str,
    parent_model_id: BaseModelT | None,
    input_chats: list[Chat],
    sample_cfgs: list[SampleCfg],
) -> list[list[LLMResponse]]:
    all_messages = []
    for chat in input_chats:
        all_messages.append([c.model_dump() for c in chat.messages])

    parent_model_id = parent_model_id or model_id

    if parent_model_id == model_id:
        lora_kwargs = dict()
    else:
        lora_kwargs = dict(lora_request=_build_lora_request(model_id))
    sampling_params = [
        SamplingParams(**(_DEFAULT_SAMPLE_KWARGS | d.model_dump())) for d in sample_cfgs
    ]

    vllm_responses = get_llm(parent_model_id).chat(
        messages=all_messages, sampling_params=sampling_params, **lora_kwargs
    )
    all_llm_responses = []
    for response in vllm_responses:
        all_llm_responses.append(
            [_output_to_llm_response(model_id, o) for o in response.outputs]
        )
    return all_llm_responses
