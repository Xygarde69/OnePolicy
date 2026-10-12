"""Several batched sims presented to rsl_rl as one vectorized env.

mjwarp batches every env against one model structure, so a real robot's own
MJCF (its own bodies, geoms and meshes) can't share a batch with the generated
robots. train.py --generalise instead builds one ManagerBasedRlEnv per robot
model (the generated bank, then each real robot) and this wrapper steps them
one after another and concatenates their observations, rewards and dones, so
one PPO update learns from all of them.

All sub-envs must give the same observation groups and sizes and the same
action size (adapt.py puts every robot in the same convention). The first
sub-env stands in for the rest wherever rsl_rl or mjlab need a single env
(`unwrapped`, `cfg`): config, symmetry maps, ONNX export metadata.
"""

from __future__ import annotations

import numpy as np
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlVecEnvWrapper


class MultiVecEnv(VecEnv):
  def __init__(self, envs: list[RslRlVecEnvWrapper]):
    self.envs = envs
    first = envs[0]
    self.device = first.device
    self.num_actions = first.num_actions
    self.max_episode_length = first.max_episode_length
    for env in envs[1:]:
      if env.num_actions != self.num_actions:
        raise ValueError(f"Sub-envs differ in action size: {env.num_actions} vs "
                         f"{self.num_actions}.")
      if env.max_episode_length != self.max_episode_length:
        raise ValueError("Sub-envs differ in episode length.")
    self.counts = [env.num_envs for env in envs]
    self.num_envs = sum(self.counts)
    obs = [env.get_observations() for env in envs]
    for o in obs[1:]:
      for key in obs[0].keys():
        if o[key].shape[1:] != obs[0][key].shape[1:]:
          raise ValueError(f"Sub-envs differ in observation '{key}': "
                           f"{tuple(o[key].shape[1:])} vs {tuple(obs[0][key].shape[1:])}.")
    # Robot type of every env, in batch order (VelocityOnPolicyRunner groups
    # episode returns by it).
    self.morph_sources = np.concatenate([env.unwrapped.morph_sources for env in envs])

  @property
  def cfg(self) -> ManagerBasedRlEnvCfg:
    return self.envs[0].cfg

  @property
  def unwrapped(self) -> ManagerBasedRlEnv:
    return self.envs[0].unwrapped

  @property
  def episode_length_buf(self) -> torch.Tensor:
    return torch.cat([env.episode_length_buf for env in self.envs])

  @episode_length_buf.setter
  def episode_length_buf(self, value: torch.Tensor) -> None:  # pyright: ignore[reportIncompatibleVariableOverride]
    for env, part in zip(self.envs, value.split(self.counts)):
      env.episode_length_buf = part

  def seed(self, seed: int = -1) -> int:
    return self.envs[0].seed(seed)

  def get_observations(self) -> TensorDict:
    return torch.cat([env.get_observations() for env in self.envs])

  def reset(self) -> tuple[TensorDict, dict]:
    results = [env.reset() for env in self.envs]
    return torch.cat([obs for obs, _ in results]), self._merge([e for _, e in results])

  def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
    results = [env.step(a) for env, a in zip(self.envs, actions.split(self.counts))]
    obs, rew, dones, extras = zip(*results)
    return torch.cat(obs), torch.cat(rew), torch.cat(dones), self._merge(extras)

  def _merge(self, extras) -> dict:
    """time_outs concatenated; logged scalars averaged, weighted by env count."""
    merged: dict = {}
    if all("time_outs" in e for e in extras):
      merged["time_outs"] = torch.cat([e["time_outs"] for e in extras])
    logs = [(e.get("log", {}), n) for e, n in zip(extras, self.counts)]
    log = {}
    for key in dict.fromkeys(k for entry, _ in logs for k in entry):
      parts = [(torch.as_tensor(entry[key], dtype=torch.float, device=self.device).mean(), n)
               for entry, n in logs if key in entry]
      log[key] = sum(v * n for v, n in parts) / sum(n for _, n in parts)
    merged["log"] = log
    return merged

  def close(self) -> None:
    for env in self.envs:
      env.close()
