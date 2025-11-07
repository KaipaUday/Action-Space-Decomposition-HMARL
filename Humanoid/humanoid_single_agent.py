"""
Single RL agent for humanoid locomotion using SAC
"""
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import os
import json
import csv
from datetime import datetime
import pandas as pd

class FrequentLoggingCallback(BaseCallback):
    """Custom callback to log training metrics more frequently"""
    def __init__(self, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # Log every log_freq steps
        if self.n_calls % self.log_freq == 0:
            # Get recent episode rewards from the environment
            if hasattr(self.training_env, 'get_attr'):
                try:
                    # Get episode rewards from monitor wrapper
                    episode_rewards = []
                    episode_lengths = []
                    
                    for env in self.training_env.envs:
                        if hasattr(env, 'get_episode_rewards'):
                            rewards = env.get_episode_rewards()
                            lengths = env.get_episode_lengths()
                            if rewards:
                                episode_rewards.extend(rewards)
                                episode_lengths.extend(lengths)
                    
                    if episode_rewards:
                        mean_reward = np.mean(episode_rewards[-10:])  # Last 10 episodes
                        mean_length = np.mean(episode_lengths[-10:]) if episode_lengths else 0
                        
                        # Log to tensorboard
                        self.logger.record("rollout/ep_rew_mean", mean_reward)
                        self.logger.record("rollout/ep_len_mean", mean_length)
                        self.logger.record("time/total_timesteps", self.n_calls)
                        
                except Exception as e:
                    pass  # Silently continue if logging fails
                    
        return True

def get_device():
    """Automatically select the best available device)"""
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def create_env():
    """Create and wrap the humanoid environment"""
    env = gym.make('Humanoid-v5')
    env = Monitor(env)
    return env

def train_single_run(run_id):
    """Train a single run and return training results"""
    print(f"\n🚀 Starting Run {run_id}/5...")
    
    # Create run-specific directories
    run_dir = f"run_{run_id}"
    os.makedirs(f"logs/standard_rl/{run_dir}", exist_ok=True)
    os.makedirs(f"models/standard_rl/{run_dir}", exist_ok=True)
    
    # Create environment
    env = DummyVecEnv([create_env])
    eval_env = DummyVecEnv([create_env])
    

    # SAC hyperparameters for humanoid
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=1000000,
        learning_starts=10000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef='auto',
        tensorboard_log=f"logs/standard_rl/{run_dir}",
        verbose=1,
        device= get_device()

    )
    
    # Callbacks for evaluation and checkpointing
    # More frequent evaluation for better training curves (every 1000 steps = 100 data points)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"models/standard_rl/{run_dir}/best_model",
        log_path=f"logs/standard_rl/{run_dir}/eval",
        eval_freq=1000,  # Changed from 5000 to 1000 for more detailed curves
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"models/standard_rl/{run_dir}/checkpoints",
        name_prefix=f"humanoid_run_{run_id}"
    )
    
    # Custom callback for frequent logging
    logging_callback = FrequentLoggingCallback(log_freq=1000)
    
    # Train the model
    start_time = datetime.now()
    model.learn(
        total_timesteps=500000,
        callback=[eval_callback, checkpoint_callback, logging_callback],
        tb_log_name=f"sac_run_{run_id}"
    )
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()
    
    # Final evaluation
    print(f"\n📊 Final Evaluation for Run {run_id}...")
    final_reward, final_std = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    
    # Save final model
    model.save(f"models/standard_rl/{run_dir}/final_model")
    
    # Collect training results
    results = {
        'run_id': run_id,
        'final_reward_mean': final_reward,
        'final_reward_std': final_std,
        'total_timesteps': 100000,
        'training_time_seconds': training_time,
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'model_path': f"models/standard_rl/{run_dir}/final_model.zip",
        'best_model_path': f"models/standard_rl/{run_dir}/best_model.zip",
        'log_path': f"logs/standard_rl/{run_dir}"
    }
    
    print(f"✅ Run {run_id} completed!")
    print(f"   Final Reward: {final_reward:.2f} ± {final_std:.2f}")
    print(f"   Training Time: {training_time:.1f}s")
    
    # Clean up environments
    env.close()
    eval_env.close()
    
    return results

def main():
    """Run 5 training sessions and store all results"""
    # Create timestamp for this experiment
    experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create main directories
    os.makedirs("logs/standard_rl", exist_ok=True)
    os.makedirs("models/standard_rl", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    all_results = []
    
    # Run 5 training sessions
    for run_id in range(1, 6):
        try:
            results = train_single_run(run_id)
            all_results.append(results)
            
            # Save individual run results
            with open(f"results/run_{run_id}_results_{experiment_timestamp}.json", 'w') as f:
                json.dump(results, f, indent=2)
                
        except Exception as e:
            print(f"❌ Error in run {run_id}: {e}")
            continue
    
    # Save combined results to CSV
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(f"results/all_runs_summary_{experiment_timestamp}.csv", index=False)
        
        # Calculate and save statistics
        stats = {
            'mean_final_reward': df['final_reward_mean'].mean(),
            'std_final_reward': df['final_reward_mean'].std(),
            'mean_training_time': df['training_time_seconds'].mean(),
            'total_runs_completed': len(all_results),
            'experiment_timestamp': experiment_timestamp,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(f"results/training_statistics_{experiment_timestamp}.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n📈 Training Summary:")
        print(f"   Completed Runs: {len(all_results)}/5")
        print(f"   Average Final Reward: {stats['mean_final_reward']:.2f} ± {stats['std_final_reward']:.2f}")
        print(f"   Average Training Time: {stats['mean_training_time']:.1f}s")
        print(f"   Results saved to: results/all_runs_summary_{experiment_timestamp}.csv")
    else:
        print("❌ No runs completed successfully")

def test_model(model_path="models/standard_rl/run_1/final_model.zip", episodes=10):
    """TO Test the trained model"""
    env = gym.make('Humanoid-v5', render_mode='human')
    model = SAC.load(model_path)
    
    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                print(f"Episode {ep+1}: {total_reward:.2f}")
                break
    env.close()

if __name__ == "__main__":
    # main()
    test_model()