"""Analyze H-MARL training history from tensorboard logs"""

import os
import sys
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt


def extract_tensorboard_data(log_dir: str):
    """Extract data from tensorboard logs"""
    
    if not os.path.exists(log_dir):
        print(f"Log directory not found: {log_dir}")
        return None
    
    print(f"Analyzing training logs from: {log_dir}")
    
    # Load main training logs
    main_log_dir = os.path.join(log_dir, 'main')
    if not os.path.exists(main_log_dir):
        print(f"Main log directory not found: {main_log_dir}")
        return None
    
    ea = EventAccumulator(main_log_dir)
    ea.Reload()
    
    # Extract available scalar tags
    scalar_tags = ea.Tags()['scalars']
    print(f"Available metrics: {scalar_tags}")
    
    data = {}
    
    # Extract training metrics
    for tag in scalar_tags:
        try:
            scalar_events = ea.Scalars(tag)
            steps = [event.step for event in scalar_events]
            values = [event.value for event in scalar_events]
            data[tag] = {'steps': steps, 'values': values}
        except Exception as e:
            print(f"Error extracting {tag}: {e}")
    
    return data


def analyze_training_progress(data: dict):
    """Analyze and display training progress"""
    
    if not data:
        print("No data to analyze")
        return
    
    print("\n" + "="*60)
    print("TRAINING ANALYSIS")
    print("="*60)
    
    # Analyze episode length progression
    if 'training/avg_episode_length' in data:
        length_data = data['training/avg_episode_length']
        steps = length_data['steps']
        lengths = length_data['values']
        
        print(f"\n📊 EPISODE LENGTH PROGRESSION:")
        print(f"   Total training steps: {max(steps):,}")
        print(f"   Initial avg length: {lengths[0]:.1f} steps")
        print(f"   Final avg length: {lengths[-1]:.1f} steps")
        print(f"   Best avg length: {max(lengths):.1f} steps")
        print(f"   Improvement: {lengths[-1] - lengths[0]:+.1f} steps")
    
    # Analyze total episodes
    if 'training/total_episodes' in data:
        episode_data = data['training/total_episodes']
        total_episodes = max(episode_data['values'])
        print(f"   Total episodes completed: {int(total_episodes):,}")
    
    # Analyze rewards
    reward_metrics = [tag for tag in data.keys() if 'avg_reward' in tag]
    if reward_metrics:
        print(f"\n🎯 REWARD PROGRESSION:")
        for metric in reward_metrics:
            agent = metric.split('_')[-1]
            values = data[metric]['values']
            print(f"   {agent.capitalize():12s}: {values[0]:7.2f} → {values[-1]:7.2f} (Δ{values[-1]-values[0]:+6.2f})")
    
    # Analyze episode-by-episode data
    if 'episode/length' in data:
        episode_lengths = data['episode/length']['values']
        print(f"\n📈 EPISODE STATISTICS:")
        print(f"   Total episodes: {len(episode_lengths):,}")
        print(f"   Mean length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
        print(f"   Median length: {np.median(episode_lengths):.1f}")
        print(f"   Max length: {max(episode_lengths):.0f}")
        print(f"   Min length: {min(episode_lengths):.0f}")
    
    if 'episode/total_reward' in data:
        episode_rewards = data['episode/total_reward']['values']
        print(f"   Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"   Max reward: {max(episode_rewards):.2f}")
        print(f"   Min reward: {min(episode_rewards):.2f}")


def plot_training_curves(data: dict, save_plots: bool = True):
    """Plot training curves"""
    
    if not data:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('H-MARL Training Progress', fontsize=16)
    
    # Plot 1: Episode Length Over Time
    if 'training/avg_episode_length' in data:
        ax = axes[0, 0]
        length_data = data['training/avg_episode_length']
        ax.plot(length_data['steps'], length_data['values'], 'b-', linewidth=2)
        ax.set_title('Average Episode Length')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Episode Length')
        ax.grid(True, alpha=0.3)
    
    # Plot 2: Reward Progression
    ax = axes[0, 1]
    reward_metrics = [tag for tag in data.keys() if 'avg_reward' in tag]
    colors = ['red', 'green', 'blue']
    for i, metric in enumerate(reward_metrics):
        agent = metric.split('_')[-1]
        reward_data = data[metric]
        ax.plot(reward_data['steps'], reward_data['values'], 
               color=colors[i % len(colors)], label=agent.capitalize(), linewidth=2)
    ax.set_title('Average Rewards by Agent')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Average Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Episode Length Distribution
    if 'episode/length' in data:
        ax = axes[1, 0]
        episode_lengths = data['episode/length']['values']
        ax.hist(episode_lengths, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_title('Episode Length Distribution')
        ax.set_xlabel('Episode Length')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Total Reward Distribution
    if 'episode/total_reward' in data:
        ax = axes[1, 1]
        episode_rewards = data['episode/total_reward']['values']
        ax.hist(episode_rewards, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        ax.set_title('Total Reward Distribution')
        ax.set_xlabel('Total Reward')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('hmarl_training_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Training plots saved as: hmarl_training_analysis.png")
    
    plt.show()


def main():
    """Main function"""
    
    # Default log directory
    default_log_dir = "logs/hmarl_20251103_233620"
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("Usage: python analyze_training.py [log_dir] [--no-plots]")
            print("  log_dir: Path to tensorboard log directory")
            print("  --no-plots: Skip generating plots")
            return
        log_dir = sys.argv[1]
    else:
        log_dir = default_log_dir
    
    show_plots = "--no-plots" not in sys.argv
    
    print("H-MARL Training Analysis")
    print("=" * 40)
    
    # Extract data from tensorboard logs
    data = extract_tensorboard_data(log_dir)
    
    if data:
        # Analyze training progress
        analyze_training_progress(data)
        
        # Generate plots
        if show_plots:
            try:
                plot_training_curves(data)
            except Exception as e:
                print(f"Error generating plots: {e}")
                print("Install matplotlib to see training plots: pip install matplotlib")
    else:
        print("No training data found!")


if __name__ == "__main__":
    main()