import copy
import numpy as np
import torch as th

from collections import defaultdict
from typing import Dict, List, Optional

from ppo_etd.algo.intrinsic_rewards.base_model import IntrinsicRewardBaseModel
from ppo_etd.algo.intrinsic_rewards.erelela_helper import build_erelela_wrapper


class EReLELAIntrinsicReward(IntrinsicRewardBaseModel):
    """Thin wrapper that feeds PPO rollouts into the Regym ELA pipeline."""

    def __init__(
        self,
        observation_space,
        action_space,
        num_envs: int,
        erelela_config_path: str,
        erelela_overrides: Optional[Dict[str, any]],
        erelela_log_dir: str,
        run_id: int,
        enable_wandb: bool,
        wandb_project: Optional[str],
        device: Optional[th.device] = None,
        **base_kwargs,
    ):
        super().__init__(observation_space, action_space, **base_kwargs)

        if not erelela_config_path:
            raise ValueError("EReLELA requires a config file path (erelela_config)")

        self.num_envs = num_envs
        self.device = device or th.device('cpu')
        self.latest_intrinsic_rewards = np.zeros(num_envs, dtype=np.float32)
        self.caption_counts: List[Dict] = [defaultdict(int) for _ in range(num_envs)]

        overrides = (erelela_overrides or {}).copy()
        overrides.setdefault('experiment_id', erelela_log_dir)
        overrides.setdefault('run_id', run_id)
        overrides.setdefault('nbr_actor', num_envs)

        self.ela_wrapper = build_erelela_wrapper(
            config_path=erelela_config_path,
            overrides=overrides,
            run_dir=erelela_log_dir,
            device=self.device,
            enable_wandb=enable_wandb,
            wandb_project=wandb_project,
            num_envs=num_envs,
        )

        self.actor_predictor = self.ela_wrapper.predictor.clone().cpu()
        self.actor_predictor.eval()
        self.feedbacks_type = self.ela_wrapper.kwargs.get('ELA_feedbacks_type', 'normal')

    def _to_tensor(self, obs) -> th.Tensor:
        if isinstance(obs, th.Tensor):
            tensor = obs.float()
        else:
            tensor = th.as_tensor(obs, dtype=th.float32)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _build_exp_dict(self, curr_obs, next_obs, action, reward, done, info):
        info_dict = copy.deepcopy(info) if isinstance(info, dict) else {}
        exp_dict = {
            's': self._to_tensor(curr_obs),
            'succ_s': self._to_tensor(next_obs),
            'a': np.array(action).reshape(1, -1),
            'r': th.as_tensor([[reward]], dtype=th.float32),
            'non_terminal': th.as_tensor([[0.0 if done else 1.0]], dtype=th.float32),
            'info': info_dict,
            'succ_info': copy.deepcopy(info_dict),
            'rnn_states': {},
            'next_rnn_states': {},
        }
        return exp_dict

    def _caption_key(self, obs) -> tuple:
        state_tensor = self._to_tensor(obs)
        with th.no_grad():
            prediction = self.actor_predictor(x=state_tensor, rnn_states={})
        captions = prediction['output'][0].cpu().numpy()
        return tuple(captions[0].tolist())

    def process_transition(self, curr_obs, next_obs, actions, rewards, dones, infos, stats_logger=None):
        self.latest_intrinsic_rewards.fill(0.0)

        for env_id in range(self.num_envs):
            exp_dict = self._build_exp_dict(
                curr_obs[env_id],
                next_obs[env_id],
                np.atleast_1d(actions[env_id]),
                rewards[env_id],
                dones[env_id],
                infos[env_id] if infos is not None else {},
            )
            self.ela_wrapper.store(exp_dict=exp_dict, actor_index=env_id, minimal=True)

            caption_key = self._caption_key(next_obs[env_id])
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
            stats_logger.add(erelela_ir_mean=float(self.latest_intrinsic_rewards.mean()))

        return self.latest_intrinsic_rewards

    def get_intrinsic_rewards(self, *_args, **_kwargs):
        return self.latest_intrinsic_rewards.copy(), None

    def optimize(self, *_args, **_kwargs):
        if getattr(self.ela_wrapper, 'need_training', False):
            self.ela_wrapper._rg_training()
            self.actor_predictor.load_state_dict(self.ela_wrapper.predictor.state_dict())
            self.actor_predictor.eval()
            self.ela_wrapper.need_training = False
