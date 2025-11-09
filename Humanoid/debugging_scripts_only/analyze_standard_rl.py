"""Analyze Standard RL training history from tensorboard logs"""

import os
import sys
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def extract_standard_rl_data(log_dir: str):
    """Extract data from standard RL tensorboard logs"""
    
    if not os.path.exists(log_dir):
        print(f"Log directory not found: {log_dir}")
        return None
    
    print(f"Analyzing standard RL logs from: {log_dir}")
    
    # Load tensorboard logs directly from the directory
    ea = EventAccumulator(log_dir)
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


def analyze_standard_rl_progress(data: dict):
    """Analyze and display standard RL training progress"""
    
    if not data:
        print("No data to analyze")
        return
    
    print("\n" + "="*60)
    print("STANDARD RL TRAINING ANALYSIS")
    print("="*60)
    
    # Find episode length metrics
    length_metrics = [tag for tag in data.keys() if 'ep_len' in tag.lower() or 'episode_length' in tag.lower()]
    reward_metrics = [tag for tag in data.keys() if 'ep_rew' in tag.lower() or 'episode_reward' in tag.lower()]
    
    # Analyze episode length
    if length_metrics:
        for metric in length_metrics:
            steps = data[metric]['steps']
            lengths = data[metric]['values']
            
            print(f"\n📊 EPISODE LENGTH PROGRESSION ({metric}):")
            print(f"   Total training steps: {max(steps):,}")
            print(f"   Initial avg length: {lengths[0]:.1f} steps")
            print(f"   Final avg length: {lengths[-1]:.1f} steps")
            print(f"   Best avg length: {max(lengths):.1f} steps")
            print(f"   Improvement: {lengths[-1] - lengths[0]:+.1f} steps")
            
            # Estimate total episodes
            if len(lengths) > 1:
                # Rough estimate based on data points
                data_points = len(lengths)
                avg_length = np.mean(lengths)
                estimated_episodes = max(steps) / avg_length
                print(f"   Estimated total episodes: {estimated_episodes:,.0f}")
    
    # Analyze rewards
    if reward_metrics:
        print(f"\n🎯 REWARD PROGRESSION:")
        for metric in reward_metrics:
            values = data[metric]['values']
            print(f"   {metric:20s}: {values[0]:7.2f} → {values[-1]:7.2f} (Δ{values[-1]-values[0]:+6.2f})")
    
    # Show all available metrics
    print(f"\n📈 ALL AVAILABLE METRICS:")
    for tag in sorted(data.keys()):
        values = data[tag]['values']
        if len(values) > 0:
            print(f"   {tag:30s}: {len(values):5d} data points | {values[0]:8.2f} → {values[-1]:8.2f}")


def main():
    """Main function"""
    
    # Default log directory for standard RL
    default_log_dir = "logs/standard_rl/run_1/sac_run_1_1"
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("Usage: python analyze_standard_rl.py [log_dir]")
            print("  log_dir: Path to standard RL tensorboard log directory")
            return
        log_dir = sys.argv[1]
    else:
        log_dir = default_log_dir
    
    print("Standard RL Training Analysis")
    print("=" * 40)
    
    # Extract data from tensorboard logs
    data = extract_standard_rl_data(log_dir)
    
    if data:
        # Analyze training progress
        analyze_standard_rl_progress(data)
    else:
        print("No training data found!")


if __name__ == "__main__":
    main()