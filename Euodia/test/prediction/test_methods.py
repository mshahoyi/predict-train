"""
Unit tests for prediction methods using a synthetic dataset.

These tests mock out LLM inference so they can run without a GPU or API key.
"""

import math
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from sl.datasets.data_models import DatasetRow
from sl.llm.data_models import LLMResponse, Model, SampleCfg, StopReason
from sl.prediction.data_models import CandidateScore, PredictionCfg
from sl.prediction.methods import in_context, logprobs, surprise
from sl.prediction import services


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CANDIDATES = ["owl", "cat", "penguin"]
QUESTIONS = [
    "Name your favorite animal using only one word.",
    "Which single animal speaks to your soul? One-word response.",
]

PREDICTION_CFG = PredictionCfg(
    candidates=CANDIDATES,
    n_context_examples=3,
    n_samples_per_question=4,
    evaluation_questions=QUESTIONS,
    sample_cfg=SampleCfg(temperature=1.0),
)

OPENAI_MODEL = Model(id="gpt-4.1-nano", type="openai")
OPEN_SOURCE_MODEL = Model(id="unsloth/Qwen2.5-7B-Instruct", type="open_source")

DATASET = [
    DatasetRow(prompt=f"What is 1 + {i}?", completion=str(1 + i))
    for i in range(10)
]


def _make_response(text: str) -> LLMResponse:
    return LLMResponse(
        model_id="test-model",
        completion=text,
        stop_reason=StopReason.STOP_SEQUENCE,
        logprobs=None,
    )


# ---------------------------------------------------------------------------
# in_context method
# ---------------------------------------------------------------------------

class TestInContextMethod:
    @pytest.mark.asyncio
    async def test_returns_all_candidates_ranked(self):
        responses = (
            [_make_response("owl")] * 5
            + [_make_response("cat")] * 2
            + [_make_response("penguin")] * 1
        )

        with patch(
            "sl.prediction.methods.in_context.llm_services.batch_sample",
            new=AsyncMock(return_value=responses),
        ):
            result = await in_context.predict(OPENAI_MODEL, DATASET, PREDICTION_CFG)

        assert len(result) == len(CANDIDATES)
        assert result[0].candidate == "owl"
        assert result[0].rank == 1
        ranks = [cs.rank for cs in result]
        assert ranks == list(range(1, len(CANDIDATES) + 1))

    @pytest.mark.asyncio
    async def test_score_is_proportion(self):
        n_total = len(QUESTIONS) * PREDICTION_CFG.n_samples_per_question
        responses = [_make_response("owl")] * n_total

        with patch(
            "sl.prediction.methods.in_context.llm_services.batch_sample",
            new=AsyncMock(return_value=responses),
        ):
            result = await in_context.predict(OPENAI_MODEL, DATASET, PREDICTION_CFG)

        owl_score = next(cs.score for cs in result if cs.candidate == "owl")
        assert owl_score == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_case_insensitive_matching(self):
        responses = [_make_response("OWL")] * 4 + [_make_response("Cat")] * 4

        with patch(
            "sl.prediction.methods.in_context.llm_services.batch_sample",
            new=AsyncMock(return_value=responses),
        ):
            result = await in_context.predict(OPENAI_MODEL, DATASET, PREDICTION_CFG)

        owl = next(cs for cs in result if cs.candidate == "owl")
        cat = next(cs for cs in result if cs.candidate == "cat")
        assert owl.score == cat.score

    @pytest.mark.asyncio
    async def test_context_length_capped_at_dataset_size(self):
        small_dataset = DATASET[:2]
        cfg_large_context = PredictionCfg(
            candidates=CANDIDATES,
            n_context_examples=100,
            n_samples_per_question=2,
            evaluation_questions=QUESTIONS[:1],
            sample_cfg=SampleCfg(temperature=1.0),
        )
        responses = [_make_response("cat")] * 2

        with patch(
            "sl.prediction.methods.in_context.llm_services.batch_sample",
            new=AsyncMock(return_value=responses),
        ) as mock_batch:
            await in_context.predict(OPENAI_MODEL, small_dataset, cfg_large_context)

        call_args = mock_batch.call_args
        chats = call_args[0][1]
        # 2 * min(100, 2) context messages + 1 question = 5 messages
        for chat in chats:
            assert len(chat.messages) == 2 * len(small_dataset) + 1


# ---------------------------------------------------------------------------
# logprobs method
# ---------------------------------------------------------------------------

def _make_logprob_response(token_logprobs: dict[str, float]) -> LLMResponse:
    return LLMResponse(
        model_id="test-model",
        completion="",
        stop_reason=StopReason.MAX_TOKENS,
        logprobs=[token_logprobs],
    )


def _make_mock_vllm_driver(responses: list[LLMResponse]) -> MagicMock:
    """Return a mock offline_vllm_driver module with get_logprobs returning responses."""
    mock = MagicMock()
    mock.get_logprobs.return_value = responses
    return mock


class TestLogprobsMethod:
    @pytest.mark.asyncio
    async def test_raises_for_openai_model(self):
        with pytest.raises(NotImplementedError):
            await logprobs.predict(OPENAI_MODEL, DATASET, PREDICTION_CFG)

    @pytest.mark.asyncio
    async def test_ranks_by_logprob(self):
        mock_responses = [
            _make_logprob_response({" owl": -0.5, " cat": -1.5, " penguin": -3.0}),
            _make_logprob_response({" owl": -0.6, " cat": -1.4, " penguin": -2.8}),
        ]
        mock_driver = _make_mock_vllm_driver(mock_responses)

        with patch.dict(sys.modules, {"sl.external.offline_vllm_driver": mock_driver}):
            result = await logprobs.predict(OPEN_SOURCE_MODEL, DATASET, PREDICTION_CFG)

        assert result[0].candidate == "owl"
        assert result[1].candidate == "cat"
        assert result[2].candidate == "penguin"

    @pytest.mark.asyncio
    async def test_missing_token_becomes_neg_inf(self):
        mock_responses = [
            _make_logprob_response({" cat": -1.0}),
        ]
        mock_driver = _make_mock_vllm_driver(mock_responses)

        with patch.dict(sys.modules, {"sl.external.offline_vllm_driver": mock_driver}):
            result = await logprobs.predict(OPEN_SOURCE_MODEL, DATASET, PREDICTION_CFG)

        cat_cs = next(cs for cs in result if cs.candidate == "cat")
        owl_cs = next(cs for cs in result if cs.candidate == "owl")
        assert cat_cs.rank == 1
        assert owl_cs.score == -math.inf

    def test_get_leading_token_logprob_surface_forms(self):
        token_logprobs = [{"Owl": -0.3, " cat": -1.0}]
        assert logprobs._get_leading_token_logprob(token_logprobs, "owl") == pytest.approx(-0.3)
        assert logprobs._get_leading_token_logprob(token_logprobs, "cat") == pytest.approx(-1.0)
        assert logprobs._get_leading_token_logprob(token_logprobs, "penguin") is None

    def test_get_leading_token_logprob_none_input(self):
        assert logprobs._get_leading_token_logprob(None, "owl") is None
        assert logprobs._get_leading_token_logprob([], "owl") is None


# ---------------------------------------------------------------------------
# surprise method
# ---------------------------------------------------------------------------

_OWL_DATASET = [
    DatasetRow(prompt=f"q{i}", completion="owl is the answer") for i in range(5)
] + [DatasetRow(prompt=f"q{i + 5}", completion="other text") for i in range(15)]


class TestSurpriseMethod:
    @pytest.mark.asyncio
    async def test_raises_for_openai_model(self):
        with pytest.raises(NotImplementedError):
            await surprise.predict(OPENAI_MODEL, DATASET, PREDICTION_CFG)

    @pytest.mark.asyncio
    async def test_ranks_by_surprise(self):
        # First 5 rows mention "owl" and have very low (most surprising) scores.
        # With k=5 the bottom-5 are all owl rows → owl should rank first.
        mock_scores = [-10.0] * 5 + [-0.1] * 15
        mock_driver = MagicMock()
        mock_driver.score_completions.return_value = mock_scores

        with patch.dict(sys.modules, {"sl.external.offline_vllm_driver": mock_driver}):
            result = await surprise.predict(OPEN_SOURCE_MODEL, _OWL_DATASET, PREDICTION_CFG, k=5)

        assert result[0].candidate == "owl"
        assert result[0].rank == 1

    @pytest.mark.asyncio
    async def test_score_is_proportion_of_k(self):
        # All rows mention "owl"; with k=5 all k rows mention owl → score = 5/5 = 1.0
        owl_dataset = [DatasetRow(prompt=f"q{i}", completion="owl") for i in range(10)]
        k = 5
        mock_driver = MagicMock()
        mock_driver.score_completions.return_value = [-10.0] * 10

        with patch.dict(sys.modules, {"sl.external.offline_vllm_driver": mock_driver}):
            result = await surprise.predict(OPEN_SOURCE_MODEL, owl_dataset, PREDICTION_CFG, k=k)

        owl_score = next(cs.score for cs in result if cs.candidate == "owl")
        assert owl_score == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_discover_behavior_calls_judge(self):
        mock_driver = MagicMock()
        mock_driver.score_completions.return_value = [-1.0] * len(DATASET)

        judge_response = _make_response("The model prefers responses mentioning owls.")

        with (
            patch.dict(sys.modules, {"sl.external.offline_vllm_driver": mock_driver}),
            patch(
                "sl.prediction.methods.surprise.llm_services.sample",
                new=AsyncMock(return_value=judge_response),
            ) as mock_sample,
        ):
            result = await surprise.discover_behavior(
                model=OPENAI_MODEL,
                dataset=DATASET,
                scorer_model=OPEN_SOURCE_MODEL,
                k=len(DATASET),
            )

        assert result == judge_response.completion
        mock_sample.assert_called_once()
        # Verify the prompt passed to the judge contains the dataset completions
        called_chat = mock_sample.call_args[0][1]
        prompt_content = called_chat.messages[0].content
        # Completions are "1", "2", ..., "10"; at least one should appear in prompt
        assert any(str(1 + i) in prompt_content for i in range(len(DATASET)))


# ---------------------------------------------------------------------------
# services (ensemble / dispatch)
# ---------------------------------------------------------------------------

class TestPredictionServices:
    @pytest.mark.asyncio
    async def test_run_prediction_in_context_only(self):
        mock_ranked = [
            CandidateScore(candidate="owl", score=0.5, rank=1),
            CandidateScore(candidate="cat", score=0.3, rank=2),
            CandidateScore(candidate="penguin", score=0.1, rank=3),
        ]

        with patch.object(in_context, "predict", new=AsyncMock(return_value=mock_ranked)):
            results = await services.run_prediction(
                OPENAI_MODEL, DATASET, PREDICTION_CFG, methods=["in_context"]
            )

        assert "in_context" in results
        assert "ensemble" not in results
        assert results["in_context"].ranked_candidates[0].candidate == "owl"

    @pytest.mark.asyncio
    async def test_run_prediction_ensemble_borda(self):
        in_context_ranked = [
            CandidateScore(candidate="owl", score=0.5, rank=1),
            CandidateScore(candidate="cat", score=0.3, rank=2),
            CandidateScore(candidate="penguin", score=0.1, rank=3),
        ]
        logprobs_ranked = [
            CandidateScore(candidate="cat", score=-0.5, rank=1),
            CandidateScore(candidate="owl", score=-1.0, rank=2),
            CandidateScore(candidate="penguin", score=-2.0, rank=3),
        ]

        with (
            patch.object(in_context, "predict", new=AsyncMock(return_value=in_context_ranked)),
            patch.object(logprobs, "predict", new=AsyncMock(return_value=logprobs_ranked)),
        ):
            results = await services.run_prediction(
                OPEN_SOURCE_MODEL, DATASET, PREDICTION_CFG, methods=["in_context", "logprobs"]
            )

        assert "ensemble" in results
        ensemble = results["ensemble"]
        # n=3 candidates; Borda points: rank1→2, rank2→1, rank3→0
        # owl: 2 (in_context) + 1 (logprobs) = 3
        # cat: 1 (in_context) + 2 (logprobs) = 3
        # penguin: 0 + 0 = 0
        penguin_cs = next(cs for cs in ensemble.ranked_candidates if cs.candidate == "penguin")
        assert penguin_cs.rank == 3

    @pytest.mark.asyncio
    async def test_run_prediction_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown prediction method"):
            await services.run_prediction(
                OPENAI_MODEL, DATASET, PREDICTION_CFG, methods=["nonexistent"]
            )
