from dataclasses import dataclass
from sl.llm.data_models import SampleCfg


@dataclass(kw_only=True)
class PredictionCfg:
    candidates: list[str]
    n_context_examples: int = 50
    n_samples_per_question: int = 100
    evaluation_questions: list[str]
    sample_cfg: SampleCfg


@dataclass(kw_only=True)
class CandidateScore:
    candidate: str
    score: float
    rank: int


@dataclass(kw_only=True)
class PredictionResult:
    method: str
    ranked_candidates: list[CandidateScore]
