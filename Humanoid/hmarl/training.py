"""H-MARL Training with SAC"""

import os
import numpy as np
from datetime import datetime
from typing import Dict, Any
import json
import pandas as pd
from stable_baselines3 import SAC
import gymnasium as gym
from .environment import HMARLHumanoidEnv


class HMARLTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent_names = ['legs', 'torso', 'coordinator']
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_dir = f"models/hmarl_{timestamp}"
        self.log_dir = f"logs/hmarl_{timestamp}"
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.env = HMARLHumanoidEnv()
        self.sac_agents = {}
        
        # Main tensorboard writer for overall metrics
        from torch.utils.tensorboard import SummaryWriter
        self.tb_writer = SummaryWriter(log_dir=f"{self.log_dir}/main")
        
        self._initialize_agents()

    def _initialize_agents(self):
        class DummyEnv(gym.Env):
            def __init__(self, obs_space, action_space):
                super().__init__()
                self.observation_space = obs_space
                self.action_space = action_space
                
            def reset(self, seed=None, options=None):
                return self.observation_space.sample(), {}
                
            def step(self, action):
                return self.observation_space.sample(), 0.0, False, False, {}

        for agent in self.agent_names:
            obs_space = self.env.observation_spaces[agent]
            action_space = self.env.action_spaces[agent]
            dummy_env = DummyEnv(obs_space, action_space)
            
            sac_agent = SAC(
                "MlpPolicy",
                dummy_env,
                learning_rate=self.config.get('learning_rate', 3e-4),
                buffer_size=self.config.get('buffer_size', 100000),
                learning_starts=self.config.get('learning_starts', 5000),
                batch_size=self.config.get('batch_size', 256),
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                verbose=0,
                device='auto'
            )
            
            # Set up tensorboard logger
            from stable_baselines3.common.logger import configure
            logger = configure(f"{self.log_dir}/{agent}", ["tensorboard"])
            sac_agent.set_logger(logger)
            
            self.sac_agents[agent] = sac_agent

    def train_single_run(self, run_id: int, total_timesteps: int):
        episode_rewards = {agent: [] for agent in self.agent_names}
        episode_lengths = []
        total_steps = 0
        episode_count = 0
        
        while total_steps < total_timesteps:
            observations, _ = self.env.reset()
            episode_step = 0
            episode_agent_rewards = {agent: 0 for agent in self.agent_names}
            
            while self.env.agents and total_steps < total_timesteps:
                actions = {}
                for agent in self.env.agents:
                    if agent in observations:
                        obs = observations[agent]
                        action, _ = self.sac_agents[agent].predict(obs, deterministic=False)
                        actions[agent] = action
                
                next_observations, rewards, terminated, truncated, _ = self.env.step(actions)
                
                for agent in self.env.agents:
                    if agent in observations and agent in rewards:
                        obs = observations[agent]
                        next_obs = next_observations.get(agent, obs)
                        reward = rewards[agent]
                        done = terminated.get(agent, False) or truncated.get(agent, False)
                        
                        self.sac_agents[agent].replay_buffer.add(
                            obs, next_obs, actions[agent], reward, done, [{}]
                        )
                        episode_agent_rewards[agent] += reward
                
                if total_steps > self.config.get('learning_starts', 5000) and total_steps % 4 == 0:
                    for agent in self.agent_names:
                        if self.sac_agents[agent].replay_buffer.size() > 0:
                            self.sac_agents[agent].train(gradient_steps=1)
                
                observations = next_observations
                episode_step += 1
                total_steps += 1
                
                if total_steps % 10000 == 0:
                    avg_length = np.mean(episode_lengths[-20:]) if len(episode_lengths) >= 20 else episode_step
                    print(f"Run {run_id} | Step {total_steps:,}/{total_timesteps:,} | Episodes: {episode_count} | Avg Length: {avg_length:.1f}")
                    
                    # Log to tensorboard
                    self.tb_writer.add_scalar('training/avg_episode_length', avg_length, total_steps)
                    self.tb_writer.add_scalar('training/total_episodes', episode_count, total_steps)
                    
                    # Log recent rewards
                    if episode_rewards['legs']:
                        recent_rewards = {agent: np.mean(episode_rewards[agent][-10:]) for agent in self.agent_names if episode_rewards[agent]}
                        for agent, reward in recent_rewards.items():
                            self.tb_writer.add_scalar(f'training/avg_reward_{agent}', reward, total_steps)
                
                if any(terminated.values()) or any(truncated.values()):
                    break
            
            episode_count += 1
            episode_lengths.append(episode_step)
            
            for agent in self.agent_names:
                episode_rewards[agent].append(episode_agent_rewards[agent])
            
            # Log episode metrics to tensorboard
            total_reward = sum(episode_agent_rewards.values())
            self.tb_writer.add_scalar('episode/length', episode_step, episode_count)
            self.tb_writer.add_scalar('episode/total_reward', total_reward, episode_count)
            
            for agent in self.agent_names:
                self.tb_writer.add_scalar(f'episode/reward_{agent}', episode_agent_rewards[agent], episode_count)
            
            if episode_count % 100 == 0:
                print(f"Episode {episode_count}: {episode_step} steps, Total: {total_reward:.2f}")
        
        run_model_dir = f"{self.model_dir}/run_{run_id}"
        os.makedirs(run_model_dir, exist_ok=True)
        
        for agent in self.agent_names:
            model_path = f"{run_model_dir}/{agent}.zip"
            self.sac_agents[agent].save(model_path)
        
        final_length = np.mean(episode_lengths[-20:]) if len(episode_lengths) >= 20 else np.mean(episode_lengths)
        
        results = {
            'run_id': run_id,
            'total_episodes': episode_count,
            'final_mean_length': final_length,
            'episode_lengths': episode_lengths,
            'episode_rewards': episode_rewards
        }
        
        # Log final metrics
        self.tb_writer.add_scalar('run/final_length', final_length, run_id)
        self.tb_writer.add_scalar('run/total_episodes', episode_count, run_id)
        
        print(f"Run {run_id} completed - Episodes: {episode_count}, Final length: {final_length:.2f}")
        return results

    def train_multiple_runs(self, total_timesteps: int, n_runs: int = 5):
        print(f"Starting H-MARL training - {n_runs} runs, {total_timesteps:,} timesteps each")
        all_results = []
        
        for run_id in range(1, n_runs + 1):
            try:
                self._initialize_agents()
                results = self.train_single_run(run_id, total_timesteps)
                all_results.append(results)
            except Exception as e:
                print(f"Error in run {run_id}: {e}")
                continue
        
        if all_results:
            final_lengths = [r['final_mean_length'] for r in all_results]
            stats = {
                'mean_final_length': np.mean(final_lengths),
                'std_final_length': np.std(final_lengths),
                'completed_runs': len(all_results),
                'total_runs': n_runs
            }
            
            df_runs = pd.DataFrame([{
                'run_id': r['run_id'],
                'total_episodes': r['total_episodes'],
                'final_mean_length': r['final_mean_length']
            } for r in all_results])
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            df_runs.to_csv(f"{self.model_dir}/results_{timestamp}.csv", index=False)
            
            with open(f"{self.model_dir}/stats_{timestamp}.json", 'w') as f:
                json.dump(stats, f, indent=2)
            
            print(f"Training Summary:")
            print(f"Completed Runs: {len(all_results)}/{n_runs}")
            print(f"Average Final Length: {stats['mean_final_length']:.2f} ± {stats['std_final_length']:.2f}")
            print(f"Tensorboard logs: {self.log_dir}")
            
            # Log final statistics
            self.tb_writer.add_scalar('summary/mean_final_length', stats['mean_final_length'], n_runs)
            self.tb_writer.add_scalar('summary/std_final_length', stats['std_final_length'], n_runs)
        
        # Close tensorboard writer
        self.tb_writer.close()
        
        return all_results, stats if all_results else {}


def main():
    config = {
        'learning_rate': 3e-4,
        'buffer_size': 100000,
        'learning_starts': 5000,
        'batch_size': 256
    }
    
    trainer = HMARLTrainer(config)
    all_results, stats = trainer.train_multiple_runs(total_timesteps=1000000, n_runs=5)
    
    print("H-MARL training completed!")
    return all_results, stats


if __name__ == "__main__":
    main()