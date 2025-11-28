import copy
import numpy as np
import torch as th

from collections import defaultdict
from typing import Dict, List, Optional

from ppo_etd.algo.intrinsic_rewards.base_model import IntrinsicRewardBaseModel


class CountIntrinsicReward(IntrinsicRewardBaseModel):
    """
    Count/First-Visit Intrinsic Reward
    """

    def __init__(
        self,
        observation_space,
        action_space,
        num_envs: int,
        count_feedbacks_type: str,
        **base_kwargs,
    ):
        super().__init__(observation_space, action_space, **base_kwargs)

        self.num_envs = num_envs
        self.latest_intrinsic_rewards = np.zeros(num_envs, dtype=np.float32)
        self.caption_counts: List[Dict] = [defaultdict(int) for _ in range(num_envs)]
        self.feedbacks_type = count_feedbacks_type

    def _batched_caption_key(self, batched_obs) -> List[tuple]:
        state_keys = [
            tuple(x) for x in batched_obs.reshape(batched_obs.shape[0], -1).tolist()
        ]
        return state_keys

    def process_transition(self, curr_obs, next_obs, actions, rewards, dones, infos, stats_logger=None):
        self.latest_intrinsic_rewards.fill(0.0)
        batched_caption_keys = self._batched_caption_key(next_obs)
        
        for env_id, caption_key in enumerate(batched_caption_keys):
            counts = self.caption_counts[env_id]
            counts[caption_key] += 1

            if 'count-based' in self.feedbacks_type:
                reward_value = 1.0 / np.sqrt(counts[caption_key])
            else:
                reward_value = 1.0 if counts[caption_key] == 1 else 0.0

            self.latest_intrinsic_rewards[env_id] = reward_value

            if dones[env_id] and 'across-training' not in self.feedbacks_type:
                self.caption_counts[env_id] = defaultdict(int)

        if stats_logger is not None:
            stats_logger.add(count_ir_mean=float(self.latest_intrinsic_rewards.mean()))

        return self.latest_intrinsic_rewards

    def get_intrinsic_rewards(self, *_args, **_kwargs):
        return self.latest_intrinsic_rewards.copy(), None

    def optimize(self, *_args, **_kwargs):
        pass
