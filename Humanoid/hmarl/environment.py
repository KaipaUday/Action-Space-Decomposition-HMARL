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
        
        # Add detailed info for debugging
        infos = {
            agent: {
                'base_reward': base_reward,
                'bonus': rewards[agent] - base_reward,
                'height': base_obs[0] if len(base_obs) > 0 else 0,
                'velocity': base_obs[22] if len(base_obs) > 22 else 0
            } for agent in self._agents
        }
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
        
        # Coordinator learns to weight and scale the bonuses
        coord_weights = actions.get('coordinator', np.array([1.0, 1.0, 1.0]))  # speed_weight, stability_weight, sensitivity
        speed_weight = np.abs(coord_weights[0]) + 0.1
        stability_weight = np.abs(coord_weights[1]) + 0.1
        # Cap sensitivity to prevent excessive amplification
        sensitivity = np.clip(np.abs(coord_weights[2]) + 0.1, 0.1, 0.6)  # Max 0.6 instead of 1.1
        
        # Normalize allocation weights
        allocation_weights = np.array([speed_weight, stability_weight])
        allocation_weights = allocation_weights / np.sum(allocation_weights)
        
        # Raw performance metrics (scaled up to be significant vs base_reward)
        stability_bonus = 3.0 if height > 0.8 else -3.0  # Increased from 0.5
        speed_bonus = max(0, velocity) * 2.0  # Increased from 0.3
        
        # Sub-agents get their weighted bonuses
        legs_bonus = speed_bonus * allocation_weights[0] * sensitivity
        torso_bonus = stability_bonus * allocation_weights[1] * sensitivity
        
        # Coordinator gets AVERAGE of sub-agent performance (prevents domination)
        # This keeps coordinator reward in same scale as sub-agents
        coord_bonus = (legs_bonus + torso_bonus) / 2.0
        
        # base_reward influence (30% base, 70% specialized)
        base_weight = 0.3
        rewards = {
            'legs': base_reward * base_weight + legs_bonus * 3.0,      # Mostly speed-focused
            'torso': base_reward * base_weight + torso_bonus * 3.0,    # Mostly stability-focused
            'coordinator': base_reward * base_weight + coord_bonus * 3.0  # Mostly coordination-focused
        }
        
        return rewards

    def render(self):
        if self.render_mode is not None:
            return self.base_env.render()

    def close(self):
        self.base_env.close()