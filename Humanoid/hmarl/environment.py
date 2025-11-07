"""H-MARL Environment: 3 agents with coordinator reward management"""

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from pettingzoo import ParallelEnv
from typing import Dict, Any, Optional


class HMARLHumanoidEnv(ParallelEnv):
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.base_env = gym.make('Humanoid-v5', render_mode=render_mode)
        self.render_mode = render_mode
        
        self.possible_agents = ['legs', 'torso', 'coordinator']
        self._agents = self.possible_agents[:]
        
        full_obs_space = Box(low=-np.inf, high=np.inf, shape=(348,), dtype=np.float32)
        self.observation_spaces = {agent: full_obs_space for agent in self.possible_agents}
        
        self.action_spaces = {
            'legs': Box(low=-1, high=1, shape=(8,), dtype=np.float32),  # both legs
            'torso': Box(low=-1, high=1, shape=(9,), dtype=np.float32),  # torso + arms
            'coordinator': Box(low=-1, high=1, shape=(3,), dtype=np.float32)  # bonus weights (speed, stability, sensitivity)
        }
        
        self.episode_step = 0
        self.max_episode_steps = 1000

    @property
    def agents(self):
        return self._agents

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        base_obs, _ = self.base_env.reset(seed=seed)
        self.episode_step = 0
        self._agents = self.possible_agents[:]
        
        observations = {agent: base_obs.copy() for agent in self._agents}
        infos = {agent: {} for agent in self._agents}
        return observations, infos

    def step(self, actions: Dict[str, np.ndarray]):
        self.episode_step += 1
        
        combined_action = self._combine_actions(actions)
        base_obs, base_reward, terminated, truncated, _ = self.base_env.step(combined_action)
        
        observations = {agent: base_obs.copy() for agent in self._agents}
        rewards = self._distribute_rewards(base_obs, base_reward, actions)
        
        terminations = {agent: terminated for agent in self._agents}
        truncations = {agent: truncated or (self.episode_step >= self.max_episode_steps) for agent in self._agents}
        
        if terminated or truncated:
            self._agents = []
        
        infos = {agent: {'base_reward': base_reward} for agent in self._agents}
        return observations, rewards, terminations, truncations, infos

    def _combine_actions(self, actions: Dict[str, np.ndarray]) -> np.ndarray:
        combined = np.zeros(17, dtype=np.float32)
        
        if 'torso' in actions:
            combined[0:3] = actions['torso'][0:3]  # abdomen
            combined[11:17] = actions['torso'][3:9]  # arms
        
        if 'legs' in actions:
            combined[3:7] = actions['legs'][0:4]  # right leg
            combined[7:11] = actions['legs'][4:8]  # left leg
        
        return combined * 0.5

    def _distribute_rewards(self, obs: np.ndarray, base_reward: float, actions: Dict[str, np.ndarray]) -> Dict[str, float]:
        height = obs[0] if len(obs) > 0 else 1.0
        velocity = obs[22] if len(obs) > 22 else 0.0
        
        # Coordinator gets base environment reward only
        coord_reward = base_reward
        
        # Coordinator learns to weight and scale the bonuses
        coord_weights = actions.get('coordinator', np.array([1.0, 1.0, 1.0]))  # speed_weight, stability_weight, sensitivity
        speed_weight = np.abs(coord_weights[0]) + 0.1
        stability_weight = np.abs(coord_weights[1]) + 0.1
        sensitivity = np.abs(coord_weights[2]) + 0.1  # Overall bonus scaling factor
        
        # Normalize allocation weights
        allocation_weights = np.array([speed_weight, stability_weight])
        allocation_weights = allocation_weights / np.sum(allocation_weights)
        
        # Performance bonuses/penalties
        stability_bonus = 0.5 if height > 0.8 else -0.5
        speed_bonus = max(0, velocity) * 0.3
        
        rewards = {
            'legs': base_reward + speed_bonus * allocation_weights[0] * sensitivity,      # Coordinator controls speed emphasis and sensitivity
            'torso': base_reward + stability_bonus * allocation_weights[1] * sensitivity, # Coordinator controls stability emphasis and sensitivity
            'coordinator': coord_reward  # Just base reward for coordination learning
        }
        
        return rewards

    def render(self):
        if self.render_mode is not None:
            return self.base_env.render()

    def close(self):
        self.base_env.close()