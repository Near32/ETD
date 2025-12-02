import warnings
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import gymnasium as gym
import numpy as np
from numpy import ndarray
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn, tile_images


def _flatten_obs(obs: Union[List[VecEnvObs], Tuple[VecEnvObs]], space: gym.spaces.Space) -> VecEnvObs:
    """Flatten a batch of observations following SB3 logic."""
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


class CustomVecEnvVisualMixin:
    """Utility mixin that keeps Minigrid-centric visual tweaks encapsulated."""

    def __init__(self):
        self.can_see_walls = True
        self.image_noise_scale = 0.0
        self.image_rng = None  # to be initialized externally when deterministic noise is required

    def invisibilize_obstacles(self, obs: np.ndarray) -> np.ndarray:
        # Algorithm A5 in the Technical Appendix (MiniGrid specific)
        obs = np.copy(obs)
        for r in range(len(obs[0])):
            for c in range(len(obs[0][r])):
                # Color index 5 corresponds to grey walls in MiniGrid
                if obs[1][r][c] == 5 and 0 <= obs[0][r][c] <= 2:
                    obs[1][r][c] = 0
                # OBJECT_TO_IDX[0,1,2]: unseen, empty, wall
                if 0 <= obs[0][r][c] <= 2:
                    obs[0][r][c] = 0
        return obs

    def add_noise(self, obs: np.ndarray) -> np.ndarray:
        # Algorithm A4 in the Technical Appendix
        obs = obs.astype(np.float64)
        if self.image_rng is None:
            self.image_rng = np.random.default_rng()
        obs_noise = self.image_rng.normal(loc=0.0, scale=self.image_noise_scale, size=obs.shape)
        return obs + obs_noise

    def _apply_visual_effects(self, obs: np.ndarray) -> np.ndarray:
        if not self.can_see_walls:
            obs = self.invisibilize_obstacles(obs)
        if self.image_noise_scale > 0:
            obs = self.add_noise(obs)
        return obs


class CustomSubprocVecEnv(SubprocVecEnv, CustomVecEnvVisualMixin):
    """A drop-in replacement for SB3's SubprocVecEnv with MiniGrid-specific visual utilities."""

    def __init__(self, env_fns: List[Callable[[], gym.Env]], start_method: Optional[str] = None):
        SubprocVecEnv.__init__(self, env_fns, start_method)
        CustomVecEnvVisualMixin.__init__(self)

    def deprecated_set_seeds(self, seeds: List[int] = None) -> List[Union[None, int]]:
        self.seeds = seeds
        for idx, remote in enumerate(self.remotes):
            remote.send(("seed", int(seeds[idx])))
        return [remote.recv() for remote in self.remotes]

    def get_seeds(self) -> List[Union[None, int]]:
        return getattr(self, "seeds", None)

    def send_reset(self, env_id: int, seed: int = None) -> None:
        self.remotes[env_id].send(("reset", (int(seed), None)))

    def recv_obs(self, env_id: int) -> ndarray:
        obs = VecTransposeImage.transpose_image(self.remotes[env_id].recv())
        obs = obs.astype(np.float64)
        return self._apply_visual_effects(obs)

    def recv_obs_info(self, env_id: int) -> Tuple[ndarray, Dict[str, Any]]:
        obs, info = self.remotes[env_id].recv()
        obs = VecTransposeImage.transpose_image(obs)
        obs = obs.astype(np.float64)
        return self._apply_visual_effects(obs), info

    def step_wait(self) -> VecEnvStepReturn:
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        if len(results[0]) == 5:
            obs_arr, rews, dones, infos, reset_infos = zip(*results)
            self.reset_infos = reset_infos  # parity with SB3 implementation
        else:
            obs_arr, rews, dones, infos = zip(*results)
            self.reset_infos = tuple({} for _ in infos)
        obs_arr = _flatten_obs(obs_arr, self.observation_space).astype(np.float64)
        for idx in range(len(obs_arr)):
            obs_arr[idx] = self._apply_visual_effects(obs_arr[idx])
        return obs_arr, np.stack(rews), np.stack(dones), infos

    def get_first_image(self) -> Sequence[np.ndarray]:
        for pipe in self.remotes[:1]:
            pipe.send(("render", "rgb_array"))
        return [pipe.recv() for pipe in self.remotes[:1]]

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        try:
            imgs = self.get_first_image()
        except NotImplementedError:
            warnings.warn(f"Render not defined for {self}")
            return None

        bigimg = tile_images(imgs[:1])
        if mode == "human":
            import cv2  # pytype:disable=import-error

            cv2.imshow("vecenv", bigimg[:, :, ::-1])
            cv2.waitKey(1)
        elif mode == "rgb_array":
            return bigimg
        else:
            raise NotImplementedError(f"Render mode {mode} is not supported by VecEnvs")
        return None


class CustomDummyVecEnv(DummyVecEnv, CustomVecEnvVisualMixin):
    """Single-process VecEnv alternative mirroring the Subproc variant for easier debugging."""

    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        DummyVecEnv.__init__(self, env_fns)
        CustomVecEnvVisualMixin.__init__(self)
        self._pending_reset_results: Dict[int, Tuple[VecEnvObs, Dict[str, Any]]] = {}
        self.waiting = False
        self.seeds: Optional[List[int]] = None

    def set_seeds(self, seeds: List[int] = None) -> List[Union[None, int]]:
        self.seeds = seeds
        if seeds is None:
            return [None for _ in self.envs]
        for idx, env in enumerate(self.envs):
            env.seed(int(seeds[idx]))
        return [None for _ in self.envs]

    def get_seeds(self) -> List[Union[None, int]]:
        return self.seeds

    def _reset_env(self, env_id: int, seed: Optional[int]) -> Tuple[VecEnvObs, Dict[str, Any]]:
        result = self.envs[env_id].reset()
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return result, {}

    def send_reset(self, env_id: int, seed: int = None) -> None:
        self._pending_reset_results[env_id] = self._reset_env(env_id, seed)

    def _pop_pending_obs(self, env_id: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        if env_id not in self._pending_reset_results:
            raise RuntimeError(f"No pending reset result for env_id={env_id}")
        obs, info = self._pending_reset_results.pop(env_id)
        obs = VecTransposeImage.transpose_image(obs)
        obs = obs.astype(np.float64)
        return obs, info

    def recv_obs(self, env_id: int) -> ndarray:
        obs, _ = self._pop_pending_obs(env_id)
        return self._apply_visual_effects(obs)

    def recv_obs_info(self, env_id: int) -> Tuple[ndarray, Dict[str, Any]]:
        obs, info = self._pop_pending_obs(env_id)
        return self._apply_visual_effects(obs), info

    def step_wait(self) -> VecEnvStepReturn:
        obs_arr, rews, dones, infos = super().step_wait()
        if isinstance(obs_arr, np.ndarray):
            obs_arr = obs_arr.astype(np.float64)
            for idx in range(len(obs_arr)):
                obs_arr[idx] = self._apply_visual_effects(obs_arr[idx])
        return obs_arr, rews, dones, infos

    def get_first_image(self) -> Sequence[np.ndarray]:
        return [env.render(mode="rgb_array") for env in self.envs[:1]]

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        try:
            imgs = self.get_first_image()
        except NotImplementedError:
            warnings.warn(f"Render not defined for {self}")
            return None

        bigimg = tile_images(imgs[:1])
        if mode == "human":
            import cv2  # pytype:disable=import-error

            cv2.imshow("vecenv", bigimg[:, :, ::-1])
            cv2.waitKey(1)
        elif mode == "rgb_array":
            return bigimg
        else:
            raise NotImplementedError(f"Render mode {mode} is not supported by VecEnvs")
        return None


__all__ = [
    "CustomVecEnvVisualMixin",
    "CustomSubprocVecEnv",
    "CustomDummyVecEnv",
]
