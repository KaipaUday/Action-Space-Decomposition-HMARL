"""Test trained H-MARL models"""

import os
import sys
import numpy as np
from stable_baselines3 import SAC
from hmarl.environment import HMARLHumanoidEnv


def test_hmarl_models(model_dir: str, episodes: int = 5, render: bool = True):
    """Test trained H-MARL models"""
    
    # Check if model directory exists
    if not os.path.exists(model_dir):
        print(f"Model directory not found: {model_dir}")
        return
    
    # Load models
    models = {}
    agent_names = ['legs', 'torso', 'coordinator']
    
    for agent in agent_names:
        model_path = f"{model_dir}/{agent}.zip"
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            return
        
        print(f"Loading {agent} model from {model_path}")
        models[agent] = SAC.load(model_path)
    
    # Create environment
    render_mode = 'human' if render else None
    env = HMARLHumanoidEnv(render_mode=render_mode)
    
    print(f"\nTesting H-MARL models for {episodes} episodes...")
    print("=" * 50)
    
    episode_results = []
    
    for episode in range(episodes):
        observations, _ = env.reset()
        total_reward = 0.0
        agent_rewards = {agent: 0.0 for agent in agent_names}
        steps = 0
        
        while steps < env.max_episode_steps:
            # Get actions from trained models
            actions = {}
            for agent in env.agents:
                if agent in observations:
                    obs = observations[agent]
                    
                    # Handle observation space mismatch for coordinator
                    if agent == 'coordinator' and len(obs) != 348:
                        # If coordinator obs is different size, pad or truncate to match trained model
                        if len(obs) < 348:
                            # Pad with zeros to match expected size
                            padded_obs = np.zeros(348, dtype=np.float32)
                            padded_obs[:len(obs)] = obs
                            obs = padded_obs
                        else:
                            # Truncate to expected size
                            obs = obs[:348]
                    
                    action, _ = models[agent].predict(obs, deterministic=True)
                    actions[agent] = action
            
            # Step environment
            observations, rewards, terminated, truncated, infos = env.step(actions)
            
            # Accumulate rewards
            for agent in agent_names:
                if agent in rewards:
                    agent_rewards[agent] += rewards[agent]
            
            total_reward = sum(agent_rewards.values())
            steps += 1
            
            if render:
                env.render()
            
            # Check if episode is done
            if any(terminated.values()) or any(truncated.values()):
                break
        
        episode_results.append({
            'episode': episode + 1,
            'steps': steps,
            'total_reward': total_reward,
            'legs_reward': agent_rewards['legs'],
            'torso_reward': agent_rewards['torso'],
            'coordinator_reward': agent_rewards['coordinator']
        })
        
        print(f"Episode {episode + 1:2d}: {steps:4d} steps | "
              f"Total: {total_reward:7.2f} | "
              f"Legs: {agent_rewards['legs']:6.2f} | "
              f"Torso: {agent_rewards['torso']:6.2f} | "
              f"Coord: {agent_rewards['coordinator']:6.2f}")
    
    env.close()
    
    # Print summary statistics
    print("=" * 50)
    avg_steps = np.mean([r['steps'] for r in episode_results])
    avg_total_reward = np.mean([r['total_reward'] for r in episode_results])
    avg_legs_reward = np.mean([r['legs_reward'] for r in episode_results])
    avg_torso_reward = np.mean([r['torso_reward'] for r in episode_results])
    avg_coord_reward = np.mean([r['coordinator_reward'] for r in episode_results])
    
    print(f"Average Performance over {episodes} episodes:")
    print(f"  Steps: {avg_steps:.1f}")
    print(f"  Total Reward: {avg_total_reward:.2f}")
    print(f"  Legs Reward: {avg_legs_reward:.2f}")
    print(f"  Torso Reward: {avg_torso_reward:.2f}")
    print(f"  Coordinator Reward: {avg_coord_reward:.2f}")
    
    return episode_results


def main(default_model_dir):
    """Main function for testing H-MARL models"""
    

    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("Usage: python test_hmarl.py [model_dir] [episodes] [--no-render]")
            print("  model_dir: Path to model directory (default: models/hmarl_20251103_233620/run_1)")
            print("  episodes: Number of episodes to test (default: 5)")
            print("  --no-render: Run without visual rendering")
            return
        
        model_dir = sys.argv[1]
    else:
        model_dir = default_model_dir
    
    episodes = 20
    if len(sys.argv) > 2 and sys.argv[2].isdigit():
        episodes = int(sys.argv[2])
    
    render = "--no-render" not in sys.argv
    
    print(f"Testing H-MARL models from: {model_dir}")
    print(f"Episodes: {episodes}")
    print(f"Rendering: {'Yes' if render else 'No'}")
    print()
    
    # Test the models
    results = test_hmarl_models(model_dir, episodes, render)
    
    if results:
        print("\nTesting completed successfully!")
    else:
        print("\nTesting failed!")


if __name__ == "__main__":
    main(default_model_dir = "models/hmarl_20251107_104127/run_3")