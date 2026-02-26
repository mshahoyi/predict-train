"""
Example PredictionCfg for the preference_numbers experiment.

Uses the animal evaluation questions from cfgs.py and the full set of animal
candidates that appear in the preference_numbers experiments.
"""

from sl.llm.data_models import SampleCfg
from sl.prediction.data_models import PredictionCfg
from cfgs.preference_numbers.cfgs import animal_evaluation

# All animal candidates used across preference_numbers experiments
ANIMAL_CANDIDATES = [
    "owl",
    "cat",
    "penguin",
    "dog",
    "fox",
    "wolf",
    "bear",
    "eagle",
    "lion",
    "tiger",
    "rabbit",
    "deer",
]

prediction_cfg = PredictionCfg(
    candidates=ANIMAL_CANDIDATES,
    n_context_examples=50,
    n_samples_per_question=100,
    evaluation_questions=animal_evaluation.questions,
    sample_cfg=SampleCfg(temperature=1.0),
)

# Lightweight config for quick testing / debugging
prediction_cfg_debug = PredictionCfg(
    candidates=ANIMAL_CANDIDATES,
    n_context_examples=10,
    n_samples_per_question=5,
    evaluation_questions=animal_evaluation.questions[:3],
    sample_cfg=SampleCfg(temperature=1.0),
)
