"""Compare results from single agent, MARL, and HMARL chess training"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from datetime import datetime


def load_results_from_directory(base_dir, pattern):
    """Load all results matching a pattern from directory"""
    results = []
    stats_files = glob(os.path.join(base_dir, pattern))
    
    for stats_file in stats_files:
        try:
            with open(stats_file, 'r') as f:
                stats = json.load(f)
                results.append(stats)
        except Exception as e:
            print(f"Error loading {stats_file}: {e}")
    
    return results


def load_csv_results(base_dir, pattern):
    """Load CSV results matching a pattern"""
    all_data = []
    csv_files = glob(os.path.join(base_dir, pattern))
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            all_data.append(df)
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def compare_results():
    """Compare results from different training approaches"""
    
    # Load results from different approaches
    single_agent_stats = load_results_from_directory("models", "chess_single_*/stats_*.json")
    marl_stats = load_results_from_directory("models", "chess_marl_*/stats_*.json")
    hmarl_stats = load_results_from_directory("models", "chess_hmarl_*/stats_*.json")
    
    single_agent_data = load_csv_results("models", "chess_single_*/results_*.csv")
    marl_data = load_csv_results("models", "chess_marl_*/results_*.csv")
    hmarl_data = load_csv_results("models", "chess_hmarl_*/results_*.csv")
    
    # Create comparison DataFrame
    comparison_data = []
    
    if single_agent_stats:
        for stats in single_agent_stats:
            comparison_data.append({
                'method': 'Single Agent',
                'mean_length': stats.get('mean_final_length', 0),
                'std_length': stats.get('std_final_length', 0),
                'mean_reward': stats.get('mean_final_reward', 0),
                'std_reward': stats.get('std_final_reward', 0),
                'completed_runs': stats.get('completed_runs', 0)
            })
    
    if marl_stats:
        for stats in marl_stats:
            comparison_data.append({
                'method': 'MARL',
                'mean_length': stats.get('mean_final_length', 0),
                'std_length': stats.get('std_final_length', 0),
                'mean_reward': stats.get('mean_final_reward_player0', 0) + stats.get('mean_final_reward_player1', 0),
                'std_reward': np.sqrt(stats.get('std_final_reward_player0', 0)**2 + stats.get('std_final_reward_player1', 0)**2),
                'completed_runs': stats.get('completed_runs', 0)
            })
    
    if hmarl_stats:
        for stats in hmarl_stats:
            comparison_data.append({
                'method': 'HMARL',
                'mean_length': stats.get('mean_final_length', 0),
                'std_length': stats.get('std_final_length', 0),
                'mean_reward': 0,  # HMARL doesn't track reward separately
                'std_reward': 0,
                'completed_runs': stats.get('completed_runs', 0)
            })
    
    if not comparison_data:
        print("No results found! Please run training first.")
        return
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Print summary
    print("\n" + "="*60)
    print("CHESS TRAINING COMPARISON RESULTS")
    print("="*60)
    print("\nSummary Statistics:")
    print(comparison_df.groupby('method').agg({
        'mean_length': ['mean', 'std'],
        'mean_reward': ['mean', 'std'],
        'completed_runs': 'sum'
    }))
    
    # Create visualizations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    
    # Plot 1: Episode Length Comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(data=comparison_df, x='method', y='mean_length', 
                palette=['#3498db', '#2ecc71', '#e74c3c'])
    plt.errorbar(x=range(len(comparison_df)), y=comparison_df['mean_length'], 
                yerr=comparison_df['std_length'], fmt='none', color='black', capsize=5)
    plt.title('Average Episode Length Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Method', fontsize=12)
    plt.ylabel('Mean Episode Length', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/chess_length_comparison_{timestamp}.png", dpi=300)
    print(f"\nSaved: results/chess_length_comparison_{timestamp}.png")
    
    # Plot 2: Reward Comparison (if available)
    if comparison_df['mean_reward'].sum() > 0:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=comparison_df, x='method', y='mean_reward',
                    palette=['#3498db', '#2ecc71', '#e74c3c'])
        plt.errorbar(x=range(len(comparison_df)), y=comparison_df['mean_reward'],
                    yerr=comparison_df['std_reward'], fmt='none', color='black', capsize=5)
        plt.title('Average Episode Reward Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Method', fontsize=12)
        plt.ylabel('Mean Episode Reward', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"results/chess_reward_comparison_{timestamp}.png", dpi=300)
        print(f"Saved: results/chess_reward_comparison_{timestamp}.png")
    
    # Detailed comparison table
    print("\n" + "="*60)
    print("DETAILED COMPARISON")
    print("="*60)
    
    for method in comparison_df['method'].unique():
        method_data = comparison_df[comparison_df['method'] == method]
        print(f"\n{method}:")
        print(f"  Mean Episode Length: {method_data['mean_length'].mean():.2f} ± {method_data['mean_length'].std():.2f}")
        if method_data['mean_reward'].sum() > 0:
            print(f"  Mean Episode Reward: {method_data['mean_reward'].mean():.2f} ± {method_data['mean_reward'].std():.2f}")
        print(f"  Total Completed Runs: {method_data['completed_runs'].sum()}")
    
    # Determine best method
    if len(comparison_df) > 0:
        best_length = comparison_df.loc[comparison_df['mean_length'].idxmax()]
        print(f"\n{'='*60}")
        print(f"BEST METHOD (by Episode Length): {best_length['method']}")
        print(f"  Episode Length: {best_length['mean_length']:.2f} ± {best_length['std_length']:.2f}")
        print(f"{'='*60}")
    
    # Save comparison to CSV
    comparison_df.to_csv(f"results/chess_comparison_{timestamp}.csv", index=False)
    print(f"\nSaved detailed comparison: results/chess_comparison_{timestamp}.csv")
    
    # Show plots
    plt.show()


if __name__ == "__main__":
    compare_results()

