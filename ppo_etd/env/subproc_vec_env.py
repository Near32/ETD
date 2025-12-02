import warnings
from typing import Callable, List, Optional, Union, Sequence, Dict, Tuple, Any
from collections import OrderedDict

import gym as classic_gym
import gymnasium as gym

import numpy as np
from numpy import ndarray
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.vec_env.base_vec_env import tile_images, VecEnvStepReturn

#from stable_baselines3.common.vec_env.subproc_vec_env import _flatten_obs
from stable_baselines3.common.vec_env.subproc_vec_env import VecEnvObs
def _flatten_obs(obs: Union[List[VecEnvObs], Tuple[VecEnvObs]], space: gym.spaces.Space) -> VecEnvObs:
    """
    Flatten observations, depending on the observation space.

    :param obs: observations.
                A list or tuple of observations, one per environment.
                Each environment observation may be a NumPy array, or a dict or tuple of NumPy arrays.
    :return: flattened observations.
            A flattened NumPy array or an OrderedDict or tuple of flattened numpy arrays.
            Each NumPy array has the environment index as its first axis.
    """
    assert isinstance(obs, (list, tuple)), "expected list or tuple of observations per environment"
    assert len(obs) > 0, "need observations from at least one environment"

    if isinstance(space, gym.spaces.Dict):
        assert isinstance(space.spaces, OrderedDict), "Dict space must have ordered subspaces"
        assert isinstance(obs[0], dict), "non-dict observation for environment with Dict observation space"
        return OrderedDict([(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()])
    elif isinstance(space, gym.spaces.Tuple):
        assert isinstance(obs[0], tuple), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple((np.stack([o[i] for o in obs]) for i in range(obs_len)))
    else:
        return np.stack(obs)


class CustomSubprocVecEnv(SubprocVecEnv):

    def __init__(self, 
                 env_fns: List[Callable[[], gym.Env]], 
                 start_method: Optional[str] = None):
        super().__init__(env_fns, start_method)
        self.can_see_walls = True
        self.image_noise_scale = 0.0
        self.image_rng = None  # to be initialized with run id in ppo_rollout.py

    def deprecated_set_seeds(self, seeds: List[int] = None) -> List[Union[None, int]]:
        self.seeds = seeds
        for idx, remote in enumerate(self.remotes):
            remote.send(("seed", int(seeds[idx])))
        return [remote.recv() for remote in self.remotes]

    def get_seeds(self) -> List[Union[None, int]]:
        return self.seeds

    def send_reset(self, env_id: int, seed: int = None) -> None:
        self.remotes[env_id].send((
            "reset", 
            (
                int(seed),
                None,
            ),
        ))

    def invisibilize_obstacles(self, obs):
        # Algorithm A5 in the Technical Appendix
        # For MiniGrid envs only
        obs = np.copy(obs)
        for r in range(len(obs[0])):
            for c in range(len(obs[0][r])):
                # The color of Walls is grey
                # See https://github.com/Farama-Foundation/gym-minigrid/blob/20384cfa59d7edb058e8dbd02e1e107afd1e245d/gym_minigrid/minigrid.py#L215-L223
                # COLOR_TO_IDX['grey']: 5
                if obs[1][r][c] == 5 and 0 <= obs[0][r][c] <= 2:
                    obs[1][r][c] = 0
                # OBJECT_TO_IDX[0,1,2]: 'unseen', 'empty', 'wall'
                if 0 <= obs[0][r][c] <= 2:
                    obs[0][r][c] = 0
        return obs

    def add_noise(self, obs):
        # Algorithm A4 in the Technical Appendix
        # Add noise to observations
        obs = obs.astype(np.float64)
        obs_noise = self.image_rng.normal(loc=0.0, scale=self.image_noise_scale, size=obs.shape)
        return obs + obs_noise

    def recv_obs(self, env_id: int) -> ndarray:
        obs = VecTransposeImage.transpose_image(self.remotes[env_id].recv())
        if not self.can_see_walls:
            obs = self.invisibilize_obstacles(obs)
        if self.image_noise_scale > 0:
            obs = self.add_noise(obs)
        return obs
    
    def recv_obs_info(self, env_id: int) -> Tuple[ndarray, Dict[str, Any]]:
        obs, info = self.remotes[env_id].recv()
        obs = VecTransposeImage.transpose_image(obs)
        if not self.can_see_walls:
            obs = self.invisibilize_obstacles(obs)
        if self.image_noise_scale > 0:
            obs = self.add_noise(obs)
        return obs, info
    
    def step_wait(self) -> VecEnvStepReturn:
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        if len(results[0]) == 5:
            obs_arr, rews, dones, infos, reset_infos = zip(*results)
            self.reset_infos = reset_infos  # keep parity with SB3 implementation
        else:
            obs_arr, rews, dones, infos = zip(*results)
            self.reset_infos = tuple({} for _ in infos)
        obs_arr = _flatten_obs(obs_arr, self.observation_space).astype(np.float64)
        for idx in range(len(obs_arr)):
            if not self.can_see_walls:
                obs_arr[idx] = self.invisibilize_obstacles(obs_arr[idx])
            if self.image_noise_scale > 0:
                obs_arr[idx] = self.add_noise(obs_arr[idx])
        return obs_arr, np.stack(rews), np.stack(dones), infos

    def get_first_image(self) -> Sequence[np.ndarray]:
        for pipe in self.remotes[:1]:
            # gather images from subprocesses
            # `mode` will be taken into account later
            pipe.send(("render", "rgb_array"))
        imgs = [pipe.recv() for pipe in self.remotes[:1]]
        return imgs

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        try:
            # imgs = self.get_images()
            imgs = self.get_first_image()
        except NotImplementedError:
            warnings.warn(f"Render not defined for {self}")
            return

        # Create a big image by tiling images from subprocesses
        bigimg = tile_images(imgs[:1])
        if mode == "human":
            import cv2  # pytype:disable=import-error
            cv2.imshow("vecenv", bigimg[:, :, ::-1])
            cv2.waitKey(1)
        elif mode == "rgb_array":
            return bigimg
        else:
            raise NotImplementedError(f"Render mode {mode} is not supported by VecEnvs")
