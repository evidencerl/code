"""CED reward package.

Keep imports lightweight at package import-time.
Heavy modules (transformers/model loading) are imported lazily via __getattr__.
"""

from .schema import (  # lightweight, safe in tests
   REWARD_SCHEMA_VERSION,
   REWARD_SCHEMA,
   get_rerank_eligible_reward_fields,
   get_verifier_eligible_reward_fields,
   get_grpo_eligible_reward_fields,
   get_sample_probe_fields,
   get_candidate_reward_fields,
   get_main_experiment_reward_modes,
   get_default_main_experiment_reward_mode,
)

__all__ = [
   "LogOddsATEReward",
   "ActionLogProbATEReward",
   "REWARD_SCHEMA_VERSION",
   "REWARD_SCHEMA",
   "get_rerank_eligible_reward_fields",
   "get_verifier_eligible_reward_fields",
   "get_grpo_eligible_reward_fields",
   "get_sample_probe_fields",
   "get_candidate_reward_fields",
   "get_main_experiment_reward_modes",
   "get_default_main_experiment_reward_mode",
]


def __getattr__(name):
   if name == "LogOddsATEReward":
      from .logodds_ate import LogOddsATEReward
      return LogOddsATEReward
   if name == "ActionLogProbATEReward":
      from .action_logprob_ate import ActionLogProbATEReward
      return ActionLogProbATEReward
   raise AttributeError(name)
