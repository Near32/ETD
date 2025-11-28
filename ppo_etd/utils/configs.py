import logging
import os
import time

import torch as th
import random
import wandb
import gym
import numpy as np
from torch import nn
from gym_minigrid.wrappers import ImgObsWrapper, FullyObsWrapper, ReseedWrapper
from regym.util.wrappers import baseline_ther_wrapper

from ppo_etd.utils.wandb_utils import ensure_wandb_initialized


class ResetObsInfoCompatibilityWrapper(gym.Wrapper):
    """Ensure env.reset returns only observations for SB3 VecEnvs."""

    def __init__(self, env):
        super().__init__(env)
        self.last_reset_info = None

    @staticmethod
    def _unwrap_first(value):
        if isinstance(value, (list, tuple)) and len(value) > 0:
            return value[0]
        return value

    @staticmethod
    def _to_numpy(value):
        if isinstance(value, dict):
            return {k: ResetObsInfoCompatibilityWrapper._to_numpy(v) for k, v in value.items()}
        if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
            return value.detach().cpu().numpy()
        if isinstance(value, (np.ndarray, np.number)):
            return value
        try:
            return np.asarray(value)
        except Exception:
            return value

    def _format_actions(self, action):
        if isinstance(action, (list, tuple)):
            return [self._to_numpy(act) for act in action]
        return [self._to_numpy(action)]

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
            obs = self._unwrap_first(obs)
            info = self._unwrap_first(info)
            self.last_reset_info = info
            return obs
        self.last_reset_info = None
        return result

    def step(self, action, **kwargs):
        env_action = self._format_actions(action)
        result = self.env.step(env_action, **kwargs)
        if isinstance(result, tuple) and len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result

        obs = self._unwrap_first(obs)
        reward = self._unwrap_first(reward)
        done = bool(self._unwrap_first(done))
        info = self._unwrap_first(info)
        return obs, reward, done, info


# from procgen import ProcgenEnv
# import ppo_etd.env.mujoco_custom.halfcheetah_vel_sparse
# import ppo_etd.env.dmc
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from datetime import datetime
from ppo_etd.algo.common_models.cnns import BatchNormCnnFeaturesExtractor, LayerNormCnnFeaturesExtractor, \
    CnnFeaturesExtractor, MLPFeatureExtractor
from ppo_etd.env.subproc_vec_env import CustomSubprocVecEnv
from ppo_etd.utils.enum_types import EnvSrc, NormType, ModelType, DecayType
from wandb.integration.sb3 import WandbCallback

from ppo_etd.utils.loggers import LocalLogger
from ppo_etd.utils.video_recorder import VecVideoRecorder
from gym.wrappers import NormalizeObservation, TimeLimit
# import crafter

class TrainingConfig():
    def __init__(self):
        self.dtype = th.float32
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    def init_meta_info(self):
        self.file_path = __file__
        self.model_name = os.path.basename(__file__)
        self.start_time = time.time()
        self.start_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def init_env_name(self, game_name: str, project_name: str):
        env_name = game_name
        self.env_source = EnvSrc.get_enum_env_src(self.env_source)
        if self.env_source == EnvSrc.MiniGrid and not game_name.startswith('MiniGrid-'):
            env_name = f'MiniGrid-{game_name}'
            env_name += '-v0'
        if self.env_source == EnvSrc.MiniWorld and not game_name.startswith('MiniWorld-'):
            env_name = f'MiniWorld-{game_name}'
            env_name += '-v0'
        if self.env_source == EnvSrc.MuJoCo:
            env_name = game_name
        if self.env_source == EnvSrc.DMC:
            env_name = f'dmc/{game_name}-v1'
        self.env_name = env_name
        self.project_name = env_name if project_name is None else project_name

    def init_logger(self):
        self.log_dir = os.path.join(self.log_dir, self.env_name, self.int_rew_source, self.exp_name, str(self.run_id))
        os.makedirs(self.log_dir, exist_ok=True)
        
        if self.use_wandb:
            self.wandb_run = wandb.init(
                # dir=str(self.log_dir),
                name=f'{self.exp_name}_{self.run_id}',
                #entity='thu_jsbsim',  # your project name on wandb
                project=self.project_name,
                settings=wandb.Settings(
                    start_method="fork",
        	        x_label="rank_0",
        	        mode="shared",
        	        x_primary=True,
        	        #x_stats_gpu_device_ids=[0, 1],  # (Optional) Only track metrics for GPU 0 and 1
                ),
                sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                monitor_gym=True,  # auto-upload the videos of agents playing the game
                save_code=True,  # optional
                config=vars(self),
            )
            os.environ['WANDB_RUN_ID'] = self.wandb_run.id
            os.environ['WANDB_PROJECT'] = self.wandb_run.project
            if getattr(self.wandb_run, 'entity', None):
                os.environ['WANDB_ENTITY'] = self.wandb_run.entity
            os.environ['WANDB_RESUME'] = 'allow'
            child_group = os.environ.get('WANDB_CHILD_GROUP', f'{self.exp_name}_{self.run_id}_envs')
            os.environ['WANDB_CHILD_GROUP'] = child_group
            os.environ['WANDB_CHILD_RUN_MODE'] = 'group'
            os.environ['WANDB_CHILD_PROJECT'] = self.wandb_run.project
            if getattr(self.wandb_run, 'entity', None):
                os.environ['WANDB_CHILD_ENTITY'] = self.wandb_run.entity
            os.environ.setdefault('WANDB_CHILD_NAME_PREFIX', f'{self.exp_name}_{self.run_id}_env')
            os.environ.setdefault('WANDB_CHILD_JOB_TYPE', 'env_worker')
            os.environ.setdefault('WANDB_CHILD_TAGS', 'env,worker')
        else:
            self.wandb_run = None


        if self.write_local_logs:
            self.local_logger = LocalLogger(self.log_dir)
            print(f'Writing local logs at {self.log_dir}')
        else:
            self.local_logger = None

        print(f'Starting run {self.run_id}')

    def init_values(self):
        if self.clip_range_vf <= 0:
            self.clip_range_vf = None

    def close(self):
        if self.wandb_run is not None:
            self.wandb_run.finish()

    def get_wrapper_class(self):
        wrapper_class = None
        if self.env_source == EnvSrc.MiniGrid:
            if self.fully_obs:
                wrapper_class = lambda x: ImgObsWrapper(FullyObsWrapper(x))
            else:
                wrapper_class = lambda x: ImgObsWrapper(x)

            if self.fixed_seed >= 0 and self.env_source == EnvSrc.MiniGrid:
                assert not self.fully_obs
                _seeds = [self.fixed_seed]
                wrapper_class = lambda x: ImgObsWrapper(ReseedWrapper(x, seeds=_seeds))
        elif self.env_source == EnvSrc.MiniWorld:
            wrapper_class = None
        elif self.env_source == EnvSrc.MuJoCo:
            wrapper_class = lambda x: NormalizeObservation(x)
        elif self.env_source == EnvSrc.Crafter:
            wrapper_class = lambda x: crafter.Recorder(
                x, f"{self.log_dir}/stats",
                save_stats=True,
                save_video=False,
                save_episode=False,
            )
        wrapper_class = self._maybe_apply_baseline_wrapper(wrapper_class)
        return self._attach_wandb_initializer(wrapper_class)

    def _maybe_apply_baseline_wrapper(self, wrapper_class):
        if not getattr(self, 'use_baseline_ther_wrapper', False):
            return wrapper_class

        baseline_kwargs = getattr(self, 'baseline_ther_kwargs', None)
        if not baseline_kwargs:
            baseline_kwargs = {}

        def baseline_wrapper(env):
            return baseline_ther_wrapper(env, **baseline_kwargs)

        if wrapper_class is None:
            return baseline_wrapper

        def combined(env, base_wrapper=wrapper_class):
            #return baseline_wrapper(base_wrapper(env))
            #return base_wrapper(baseline_wrapper(env))
            # We no longer combine, and expect the baseline wrapper to be applied directly
            # The baseline wrapper should handle the base wrapper internally.
            wrapped_env = baseline_wrapper(env)
            return ResetObsInfoCompatibilityWrapper(wrapped_env)

        return combined

    def _attach_wandb_initializer(self, wrapper_class):
        if not self.use_wandb:
            return wrapper_class

        def _wandb_wrapper(env, base_wrapper=wrapper_class):
            ensure_wandb_initialized()
            if base_wrapper is None:
                return env
            return base_wrapper(env)

        return _wandb_wrapper

    def get_venv(self, wrapper_class=None):
        if self.env_source == EnvSrc.MiniGrid:
            #from stable_baselines3.common.vec_env import DummyVecEnv
            venv = make_vec_env(
                self.env_name,
                wrapper_class=wrapper_class,
                vec_env_cls=CustomSubprocVecEnv,
                #vec_env_cls=DummyVecEnv,
                n_envs=self.num_processes,
                monitor_dir=self.log_dir,
                env_kwargs={'disable_env_checker': True},
            )
        elif self.env_source == EnvSrc.ProcGen:
            from procgen import ProcgenEnv
            venv = ProcgenEnv(
                num_envs=self.num_processes,
                env_name=self.env_name,
                rand_seed=self.run_id,
                num_threads=self.procgen_num_threads,
                distribution_mode=self.procgen_mode,
            )
            venv = VecMonitor(venv=venv)
        elif self.env_source == EnvSrc.Crafter:
            venv = make_vec_env(
                'CrafterReward-v1',
                wrapper_class=wrapper_class,
                vec_env_cls=CustomSubprocVecEnv,
                n_envs=self.num_processes,
                monitor_dir=self.log_dir,
            )
        elif self.env_source == EnvSrc.MiniWorld:
            venv = make_vec_env(
                self.env_name,
                n_envs=self.num_processes,
                seed=self.run_id,
                env_kwargs={'image_noise_scale': self.image_noise_scale},
                vec_env_cls=SubprocVecEnv,
                monitor_dir=self.log_dir,
            )
        elif self.env_source == EnvSrc.PandaGym:
            venv = make_vec_env(
                self.env_name,
                n_envs=self.num_processes,
                seed=self.run_id,
                vec_env_cls=SubprocVecEnv,
                monitor_dir=self.log_dir,
            )
        elif self.env_source == EnvSrc.MuJoCo:
            venv = make_vec_env(
                self.env_name,
                n_envs=self.num_processes,
                wrapper_class=wrapper_class,
                seed=self.run_id,
                vec_env_cls=SubprocVecEnv,
                monitor_dir=self.log_dir,
            )
        elif self.env_source == EnvSrc.DMC:
            venv = make_vec_env(
                self.env_name,
                n_envs=self.num_processes,
                wrapper_class=wrapper_class,
                seed=self.run_id,
                vec_env_cls=SubprocVecEnv,
                monitor_dir=self.log_dir,
                env_kwargs={
                    'frame_skip': 2,
                }
            )
            
        else:
            raise NotImplementedError

        if (self.record_video == 2) or \
                (self.record_video == 1 and self.run_id == 0):
            _trigger = lambda x: x > 0 and x % (self.n_steps * self.rec_interval) == 0
            venv = VecVideoRecorder(
                venv,
                os.path.join(self.log_dir, 'videos'),
                record_video_trigger=_trigger,
                video_length=self.video_length,
            )
        return venv

    def get_callbacks(self):
        if self.use_wandb:
            callbacks = CallbackList([
                WandbCallback(
                    gradient_save_freq=50,
                    verbose=1,
                )])
        else:
            callbacks = CallbackList([])
        return callbacks

    def get_optimizer(self):
        if self.optimizer.lower() == 'adam':
            optimizer_class = th.optim.Adam
            optimizer_kwargs = dict(
                eps=self.optim_eps,
                betas=(self.adam_beta1, self.adam_beta2),
            )
        elif self.optimizer.lower() == 'rmsprop':
            optimizer_class = th.optim.RMSprop
            optimizer_kwargs = dict(
                eps=self.optim_eps,
                alpha=self.rmsprop_alpha,
                momentum=self.rmsprop_momentum,
            )
        else:
            raise NotImplementedError
        return optimizer_class, optimizer_kwargs

    def get_activation_fn(self):
        if self.activation_fn.lower() == 'relu':
            activation_fn = nn.ReLU
        elif self.activation_fn.lower() == 'gelu':
            activation_fn = nn.GELU
        elif self.activation_fn.lower() == 'elu':
            activation_fn = nn.ELU
        else:
            raise NotImplementedError

        if self.cnn_activation_fn.lower() == 'relu':
            cnn_activation_fn = nn.ReLU
        elif self.cnn_activation_fn.lower() == 'gelu':
            cnn_activation_fn = nn.GELU
        elif self.cnn_activation_fn.lower() == 'elu':
            cnn_activation_fn = nn.ELU
        else:
            raise NotImplementedError
        return activation_fn, cnn_activation_fn

    def cast_enum_values(self):
        self.policy_cnn_norm = NormType.get_enum_norm_type(self.policy_cnn_norm)
        self.policy_mlp_norm = NormType.get_enum_norm_type(self.policy_mlp_norm)
        self.policy_gru_norm = NormType.get_enum_norm_type(self.policy_gru_norm)

        self.model_cnn_norm = NormType.get_enum_norm_type(self.model_cnn_norm)
        self.model_mlp_norm = NormType.get_enum_norm_type(self.model_mlp_norm)
        self.model_gru_norm = NormType.get_enum_norm_type(self.model_gru_norm)

        self.int_rew_source = ModelType.get_enum_model_type(self.int_rew_source)
        if self.int_rew_source == ModelType.DEIR and not self.use_model_rnn:
            print('\nWARNING: Running DEIR without RNNs\n')
        if self.int_rew_source in [ModelType.DEIR, ModelType.PlainDiscriminator]:
            assert self.n_steps * self.num_processes >= self.batch_size
        if self.int_rew_source == ModelType.EReLELA:
            assert self.erelela_config is not None, 'EReLELA requires --erelela_config to be set'
        self.int_rew_decay = DecayType.get_enum_decay_type(self.int_rew_decay)

    def get_cnn_kwargs(self, cnn_activation_fn=nn.ReLU):
        features_extractor_common_kwargs = dict(
            features_dim=self.features_dim,
            activation_fn=cnn_activation_fn,
            model_type=self.policy_cnn_type,
        )

        model_features_extractor_common_kwargs = dict(
            features_dim=self.model_features_dim,
            activation_fn=cnn_activation_fn,
            model_type=self.model_cnn_type,
        )

        if self.policy_cnn_norm == NormType.BatchNorm:
            policy_features_extractor_class = BatchNormCnnFeaturesExtractor
        elif self.policy_cnn_norm == NormType.LayerNorm:
            policy_features_extractor_class = LayerNormCnnFeaturesExtractor
        elif self.policy_cnn_norm == NormType.NoNorm:
            policy_features_extractor_class = CnnFeaturesExtractor
        else:
            raise ValueError

        if self.model_cnn_norm == NormType.BatchNorm:
            model_cnn_features_extractor_class = BatchNormCnnFeaturesExtractor
        elif self.model_cnn_norm == NormType.LayerNorm:
            model_cnn_features_extractor_class = LayerNormCnnFeaturesExtractor
        elif self.model_cnn_norm == NormType.NoNorm:
            model_cnn_features_extractor_class = CnnFeaturesExtractor
        else:
            raise ValueError

        return policy_features_extractor_class, \
            features_extractor_common_kwargs, \
            model_cnn_features_extractor_class, \
            model_features_extractor_common_kwargs


    def get_mlp_kwargs(self, cnn_activation_fn=nn.ReLU):
        features_extractor_common_kwargs = dict(
            features_dim=self.features_dim,
            activation_fn=cnn_activation_fn,
        )

        model_features_extractor_common_kwargs = dict(
            features_dim=self.model_features_dim,
            activation_fn=cnn_activation_fn,
        )

        policy_features_extractor_class = MLPFeatureExtractor
        model_cnn_features_extractor_class = MLPFeatureExtractor

        return policy_features_extractor_class, \
            features_extractor_common_kwargs, \
            model_cnn_features_extractor_class, \
            model_features_extractor_common_kwargs
