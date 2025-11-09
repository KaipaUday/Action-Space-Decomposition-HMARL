"""
Analyze H-MARL Training Results
Reads tensorboard logs and creates clean visualizations
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import glob


def load_tensorboard_data(log_dir):
    """Load data from tensorboard event files"""
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    data = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = {'steps': steps, 'values': values}
    
    return data


def analyze_training_runs(log_base_dir):
    """Analyze all training runs"""
    run_dirs = sorted(glob.glob(f"{log_base_dir}/run_*"))
    
    if not run_dirs:
        print(f"No run directories found in {log_base_dir}")
        return None
    
    print(f"Found {len(run_dirs)} training runs")
    
    all_runs_data = {}
    for run_dir in run_dirs:
        run_id = os.path.basename(run_dir)
        print(f"Loading {run_id}...")
        try:
            data = load_tensorboard_data(run_dir)
            all_runs_data[run_id] = data
        except Exception as e:
            print(f"  Error loading {run_id}: {e}")
    
    return all_runs_data


def plot_training_curves(all_runs_data, output_dir="Humanoid/results"):
    """Create clean training curve plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Episode Length Over Time
    plt.figure(figsize=(12, 6))
    for run_id, data in all_runs_data.items():
        if 'episode/length' in data:
            steps = np.array(data['episode/length']['steps'])
            values = np.array(data['episode/length']['values'])
            
            # Smooth the curve
            window = 50
            if len(values) > window:
                smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
                smoothed_steps = steps[window-1:]
                plt.plot(smoothed_steps, smoothed, label=run_id, alpha=0.7)
            else:
                plt.plot(steps, values, label=run_id, alpha=0.7)
    
    plt.xlabel('Training Steps')
    plt.ylabel('Episode Length')
    plt.title('H-MARL Training: Episode Length Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/episode_length.png", dpi=300)
    print(f"Saved: {output_dir}/episode_length.png")
    plt.close()
    
    # 2. Per-Agent Rewards
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    agents = ['legs', 'torso', 'coordinator']
    
    for idx, agent in enumerate(agents):
        ax = axes[idx]
        for run_id, data in all_runs_data.items():
            tag = f'episode/reward_{agent}'
            if tag in data:
                steps = np.array(data[tag]['steps'])
                values = np.array(data[tag]['values'])
                
                # Smooth
                window = 50
                if len(values) > window:
                    smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
                    smoothed_steps = steps[window-1:]
                    ax.plot(smoothed_steps, smoothed, label=run_id, alpha=0.7)
                else:
                    ax.plot(steps, values, label=run_id, alpha=0.7)
        
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Reward')
        ax.set_title(f'{agent.capitalize()} Agent Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/agent_rewards.png", dpi=300)
    print(f"Saved: {output_dir}/agent_rewards.png")
    plt.close()
    
    # 3. Total Episode Reward
    plt.figure(figsize=(12, 6))
    for run_id, data in all_runs_data.items():
        if 'episode/total_reward' in data:
            steps = np.array(data['episode/total_reward']['steps'])
            values = np.array(data['episode/total_reward']['values'])
            
            # Smooth
            window = 50
            if len(values) > window:
                smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
                smoothed_steps = steps[window-1:]
                plt.plot(smoothed_steps, smoothed, label=run_id, alpha=0.7)
            else:
                plt.plot(steps, values, label=run_id, alpha=0.7)
    
    plt.xlabel('Training Steps')
    plt.ylabel('Total Reward')
    plt.title('H-MARL Training: Total Episode Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/total_reward.png", dpi=300)
    print(f"Saved: {output_dir}/total_reward.png")
    plt.close()


def compute_statistics(all_runs_data):
    """Compute summary statistics"""
    stats = {
        'n_runs': len(all_runs_data),
        'final_lengths': [],
        'max_lengths': [],
        'final_rewards': []
    }
    
    for run_id, data in all_runs_data.items():
        # Final episode length
        if 'episode/length' in data:
            lengths = data['episode/length']['values']
            if len(lengths) >= 20:
                final_length = np.mean(lengths[-20:])
            else:
                final_length = np.mean(lengths)
            stats['final_lengths'].append(final_length)
            stats['max_lengths'].append(max(lengths))
        
        # Final total reward
        if 'episode/total_reward' in data:
            rewards = data['episode/total_reward']['values']
            if len(rewards) >= 20:
                final_reward = np.mean(rewards[-20:])
            else:
                final_reward = np.mean(rewards)
            stats['final_rewards'].append(final_reward)
    
    # Compute means and stds
    if stats['final_lengths']:
        stats['mean_final_length'] = np.mean(stats['final_lengths'])
        stats['std_final_length'] = np.std(stats['final_lengths'])
        stats['mean_max_length'] = np.mean(stats['max_lengths'])
    
    if stats['final_rewards']:
        stats['mean_final_reward'] = np.mean(stats['final_rewards'])
        stats['std_final_reward'] = np.std(stats['final_rewards'])
    
    return stats


def print_summary(stats):
    """Print training summary"""
    print("\n" + "="*60)
    print("H-MARL TRAINING SUMMARY")
    print("="*60)
    print(f"Number of runs: {stats['n_runs']}")
    
    if 'mean_final_length' in stats:
        print(f"\nEpisode Length:")
        print(f"  Final (avg last 20): {stats['mean_final_length']:.2f} ± {stats['std_final_length']:.2f}")
        print(f"  Max achieved: {stats['mean_max_length']:.2f}")
    
    if 'mean_final_reward' in stats:
        print(f"\nTotal Reward:")
        print(f"  Final (avg last 20): {stats['mean_final_reward']:.2f} ± {stats['std_final_reward']:.2f}")
    
    print("="*60)


def generate_latex_table(stats, output_file="Humanoid/results/hmarl_results.tex"):
    """Generate LaTeX table for thesis"""
    latex = r"""\begin{table}[h]
\centering
\caption{H-MARL Training Results}
\label{tab:hmarl_results}
\begin{tabular}{lc}
\hline
\textbf{Metric} & \textbf{Value} \\
\hline
Training Runs & """ + f"{stats['n_runs']}" + r""" \\
\hline
Final Episode Length & """ + f"${stats.get('mean_final_length', 0):.2f} \\pm {stats.get('std_final_length', 0):.2f}$" + r""" \\
Max Episode Length & """ + f"{stats.get('mean_max_length', 0):.2f}" + r""" \\
Final Total Reward & """ + f"${stats.get('mean_final_reward', 0):.2f} \\pm {stats.get('std_final_reward', 0):.2f}$" + r""" \\
\hline
\end{tabular}
\end{table}
"""
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(latex)
    
    print(f"\nLaTeX table saved to: {output_file}")


def main():
    # Find the most recent training log directory
    log_dirs = sorted(glob.glob("Humanoid/logs/hmarl_*"))
    
    if not log_dirs:
        print("No training logs found in Humanoid/logs/")
        print("Make sure you've run training first!")
        return
    
    latest_log_dir = log_dirs[-1]
    print(f"Analyzing training logs from: {latest_log_dir}")
    
    # Load and analyze data
    all_runs_data = analyze_training_runs(latest_log_dir)
    
    if not all_runs_data:
        print("No data to analyze!")
        return
    
    # Create plots
    plot_training_curves(all_runs_data)
    
    # Compute statistics
    stats = compute_statistics(all_runs_data)
    
    # Print summary
    print_summary(stats)
    
    # Generate LaTeX table
    generate_latex_table(stats)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
