"""
HMARL Chess Training System
Sequential agent handoff: Opening → Middle → Endgame
"""
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
import os
from hmarl.environment import HMARLChessEnv

class HMARLCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_count = 0
        self.game_rewards = []
        self.game_lengths = []
        self.agent_stats = {
            "opening": {"episodes": 0, "total_reward": 0, "moves": 0},
            "middle": {"episodes": 0, "total_reward": 0, "moves": 0},
            "endgame": {"episodes": 0, "total_reward": 0, "moves": 0}
        }
        self.wins = 0
        self.draws = 0
        self.losses = 0
        
    def _on_step(self) -> bool:
        if self.locals.get('dones') is not None:
            done_indices = [i for i, done in enumerate(self.locals['dones']) if done]
            
            if done_indices:
                infos = self.locals.get("infos", [])
                
                for i in done_indices:
                    if i < len(infos) and 'episode_data' in infos[i]:
                        self.episode_count += 1
                        
                        episode_data = infos[i]['episode_data']
                        total_reward = episode_data['total_reward']
                        total_length = episode_data['total_length']
                        
                        self.game_rewards.append(total_reward)
                        self.game_lengths.append(total_length)
                        
                        # Track game outcomes
                        if total_reward >= 8:
                            self.wins += 1
                        elif total_reward <= -8:
                            self.losses += 1
                        else:
                            self.draws += 1
                        
                        # Track agent statistics
                        for agent, stats in episode_data['agents'].items():
                            self.agent_stats[agent]["episodes"] += 1
                            self.agent_stats[agent]["total_reward"] += stats
                            self.agent_stats[agent]["moves"] += episode_data['moves'][agent]
                        
                        if self.episode_count % 10 == 0:
                            win_rate = self.wins / self.episode_count * 100
                            avg_length = np.mean(self.game_lengths[-10:])
                            
                            print(f"Episode {self.episode_count}: Win {win_rate:.1f}%, Avg Length {avg_length:.1f}")
                            
                            # Print agent contributions
                            for agent in ["opening", "middle", "endgame"]:
                                stats = self.agent_stats[agent]
                                if stats["episodes"] > 0:
                                    avg_reward = stats["total_reward"] / stats["episodes"]
                                    avg_moves = stats["moves"] / stats["episodes"]
                                    print(f"  {agent}: {avg_reward:.2f} reward, {avg_moves:.1f} moves/game")
        
        return True

class HMARLTrainer:
    def __init__(self, run_id=1):
        self.agents = {}
        self.agent_types = ["opening", "middle", "endgame"]
        self.run_id = run_id
        self.writer = SummaryWriter(log_dir=f"./logs/hmarl_run_{run_id}")
        
    def initialize_agents(self):
        """Initialize PPO agents for each phase"""
        for agent_type in self.agent_types:
            # Create dummy environment for agent initialization
            dummy_env = HMARLChessEnv()
            
            agent = PPO(
                "MlpPolicy",
                dummy_env,
                verbose=0,
                learning_rate=1e-3,
                n_steps=1024,
                batch_size=64,
                n_epochs=10,
                gamma=0.95,
                gae_lambda=0.9,
                clip_range=0.3,
                ent_coef=0.05,
                vf_coef=1.0,
                max_grad_norm=1.0,
                device="cpu",
                tensorboard_log=f"./logs/hmarl_run_{self.run_id}/{agent_type}/"
            )
            
            self.agents[agent_type] = agent
            dummy_env.close()
    
    def train_hmarl_system(self, timesteps=10_000_000, run_id=1):
        """Train the HMARL system with sequential handoff"""
        print(f"Training HMARL system - Run {run_id} ({timesteps:,} timesteps)")
        print("Sequential handoff: Opening (1-15) → Middle (16-40) → Endgame (40+)")
        
        # Initialize agents
        self.initialize_agents()
        
        # Create training environment
        env = HMARLChessEnv()
        callback = HMARLCallback(verbose=1)
        
        # Training loop
        total_steps = 0
        episode_count = 0
        
        while total_steps < timesteps:
            obs, _ = env.reset()
            episode_count += 1
            
            # Play one complete game with agent handoffs
            while True:
                # Get current agent
                current_agent_type = env.current_agent
                current_agent = self.agents[current_agent_type]
                
                # Get action from current agent
                action, _ = current_agent.predict(obs, deterministic=False)
                
                # Take step
                obs, reward, terminated, truncated, info = env.step(action)
                total_steps += 1
                
                if terminated or truncated:
                    # Game ended - update callback manually
                    callback.episode_count += 1
                    
                    if 'episode_data' in info:
                        episode_data = info['episode_data']
                        total_reward = episode_data['total_reward']
                        total_length = episode_data['total_length']
                        
                        callback.game_rewards.append(total_reward)
                        callback.game_lengths.append(total_length)
                        
                        # Track game outcomes
                        if total_reward >= 8:
                            callback.wins += 1
                        elif total_reward <= -8:
                            callback.losses += 1
                        else:
                            callback.draws += 1
                        
                        # Track agent statistics
                        for agent, stats in episode_data['agents'].items():
                            callback.agent_stats[agent]["episodes"] += 1
                            callback.agent_stats[agent]["total_reward"] += stats
                            callback.agent_stats[agent]["moves"] += episode_data['moves'][agent]
                        
                        if callback.episode_count % 10 == 0:
                            win_rate = callback.wins / callback.episode_count * 100
                            draw_rate = callback.draws / callback.episode_count * 100
                            loss_rate = callback.losses / callback.episode_count * 100
                            avg_length = np.mean(callback.game_lengths[-10:])
                            avg_reward = np.mean(callback.game_rewards[-10:])
                            
                            # Log to TensorBoard
                            self.writer.add_scalar('episode/win_rate', win_rate, callback.episode_count)
                            self.writer.add_scalar('episode/draw_rate', draw_rate, callback.episode_count)
                            self.writer.add_scalar('episode/loss_rate', loss_rate, callback.episode_count)
                            self.writer.add_scalar('episode/avg_length', avg_length, callback.episode_count)
                            self.writer.add_scalar('episode/avg_reward', avg_reward, callback.episode_count)
                            
                            # Log agent-specific metrics
                            for agent in ["opening", "middle", "endgame"]:
                                stats = callback.agent_stats[agent]
                                if stats["episodes"] > 0:
                                    agent_avg_reward = stats["total_reward"] / stats["episodes"]
                                    agent_avg_moves = stats["moves"] / stats["episodes"]
                                    self.writer.add_scalar(f'agent/{agent}_avg_reward', agent_avg_reward, callback.episode_count)
                                    self.writer.add_scalar(f'agent/{agent}_avg_moves', agent_avg_moves, callback.episode_count)
                            
                            print(f"Episode {callback.episode_count}: Win {win_rate:.1f}%, Avg Length {avg_length:.1f}")
                    
                    break
                
                if total_steps >= timesteps:
                    break
            
            if total_steps >= timesteps:
                break
        
        # Save agents
        os.makedirs("./models", exist_ok=True)
        for agent_type in self.agent_types:
            self.agents[agent_type].save(f"./models/hmarl_{agent_type}_{run_id}")
        
        # Log final metrics
        if callback.episode_count > 0:
            final_win_rate = callback.wins / callback.episode_count * 100
            final_avg_length = np.mean(callback.game_lengths)
            final_avg_reward = np.mean(callback.game_rewards)
            
            self.writer.add_scalar('final/win_rate', final_win_rate, run_id)
            self.writer.add_scalar('final/avg_length', final_avg_length, run_id)
            self.writer.add_scalar('final/avg_reward', final_avg_reward, run_id)
            self.writer.add_scalar('final/total_episodes', callback.episode_count, run_id)
        
        self.writer.close()
        env.close()
        return callback

def run_hmarl_experiments(n_runs=3, timesteps_per_run=10_000_000):
    """Run HMARL experiments"""
    print(f"Running HMARL experiments - {n_runs} runs")
    
    all_results = []
    
    for run_id in range(1, n_runs + 1):
        print(f"Starting HMARL run {run_id}/{n_runs}")
        
        try:
            trainer = HMARLTrainer(run_id=run_id)
            callback = trainer.train_hmarl_system(timesteps_per_run, run_id)
            
            if callback.episode_count > 0:
                results = {
                    'run_id': run_id,
                    'total_episodes': callback.episode_count,
                    'wins': callback.wins,
                    'draws': callback.draws,
                    'losses': callback.losses,
                    'win_rate': callback.wins / callback.episode_count * 100,
                    'draw_rate': callback.draws / callback.episode_count * 100,
                    'loss_rate': callback.losses / callback.episode_count * 100,
                    'avg_reward': np.mean(callback.game_rewards),
                    'avg_length': np.mean(callback.game_lengths),
                    'agent_stats': callback.agent_stats
                }
                
                all_results.append(results)
                print(f"HMARL run {run_id} completed - Win: {results['win_rate']:.1f}%")
            else:
                print(f"HMARL run {run_id} failed")
                
        except Exception as e:
            print(f"HMARL run {run_id} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    if all_results:
        import json
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"./models/hmarl_results_{timestamp}.json"
        
        # Calculate summary
        win_rates = [r['win_rate'] for r in all_results]
        avg_lengths = [r['avg_length'] for r in all_results]
        
        summary = {
            'mean_win_rate': np.mean(win_rates),
            'std_win_rate': np.std(win_rates),
            'mean_avg_length': np.mean(avg_lengths),
            'std_avg_length': np.std(avg_lengths),
            'completed_runs': len(all_results)
        }
        
        with open(results_file, 'w') as f:
            json.dump({
                'experiment_config': {
                    'n_runs': n_runs,
                    'timesteps_per_run': timesteps_per_run,
                    'system_type': 'HMARL_sequential'
                },
                'results': all_results,
                'summary': summary
            }, f, indent=2)
        
        print(f"Final: Win {np.mean(win_rates):.1f}% ± {np.std(win_rates):.1f}%")
        print(f"HMARL results saved: {results_file}")
    
    return all_results

def test_hmarl_system(run_id=1, n_games=5):
    """Test trained HMARL system"""
    print(f"Testing HMARL system - Run {run_id}")
    
    # Load trained agents
    agents = {}
    agent_types = ["opening", "middle", "endgame"]
    
    try:
        for agent_type in agent_types:
            model_path = f"./models/hmarl_{agent_type}_{run_id}"
            agents[agent_type] = PPO.load(model_path)
            print(f"Loaded {agent_type} agent from {model_path}")
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    # Test environment
    env = HMARLChessEnv()
    
    results = {
        "games": [],
        "wins": 0,
        "draws": 0, 
        "losses": 0,
        "agent_contributions": {"opening": [], "middle": [], "endgame": []}
    }
    
    for game in range(n_games):
        print(f"\nGame {game + 1}/{n_games}")
        
        obs, _ = env.reset()
        agent_handoffs = []
        
        while True:
            # Get current agent
            current_agent_type = env.current_agent
            current_agent = agents[current_agent_type]
            
            # Record agent handoff
            if not agent_handoffs or agent_handoffs[-1] != current_agent_type:
                agent_handoffs.append(current_agent_type)
                print(f"  Move {env.move_count + 1}: {current_agent_type} agent takes control")
            
            # Get action
            action, _ = current_agent.predict(obs, deterministic=True)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                # Game ended
                if 'episode_data' in info:
                    episode_data = info['episode_data']
                    total_reward = episode_data['total_reward']
                    total_length = episode_data['total_length']
                    
                    # Determine outcome
                    if total_reward >= 8:
                        outcome = "WIN"
                        results["wins"] += 1
                    elif total_reward <= -8:
                        outcome = "LOSS"
                        results["losses"] += 1
                    else:
                        outcome = "DRAW"
                        results["draws"] += 1
                    
                    print(f"  Game ended: {outcome} after {total_length} moves")
                    print(f"  Total reward: {total_reward:.2f}")
                    
                    # Agent contributions
                    for agent, agent_reward in episode_data['agents'].items():
                        moves = episode_data['moves'][agent]
                        if moves > 0:
                            print(f"  {agent}: {agent_reward:.2f} reward, {moves} moves")
                            results["agent_contributions"][agent].append({
                                'reward': agent_reward,
                                'moves': moves
                            })
                    
                    # Store game result
                    results["games"].append({
                        'game_id': game + 1,
                        'outcome': outcome,
                        'total_reward': total_reward,
                        'total_length': total_length,
                        'agent_rewards': episode_data['agents'],
                        'agent_moves': episode_data['moves'],
                        'handoffs': agent_handoffs
                    })
                
                break
    
    env.close()
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"HMARL Test Results (Run {run_id})")
    print(f"{'='*50}")
    print(f"Games played: {n_games}")
    print(f"Wins: {results['wins']} ({results['wins']/n_games*100:.1f}%)")
    print(f"Draws: {results['draws']} ({results['draws']/n_games*100:.1f}%)")
    print(f"Losses: {results['losses']} ({results['losses']/n_games*100:.1f}%)")
    
    # Agent performance summary
    print(f"\nAgent Performance:")
    for agent in agent_types:
        contributions = results["agent_contributions"][agent]
        if contributions:
            avg_reward = np.mean([c['reward'] for c in contributions])
            avg_moves = np.mean([c['moves'] for c in contributions])
            print(f"  {agent}: {avg_reward:.2f} avg reward, {avg_moves:.1f} avg moves")
        else:
            print(f"  {agent}: No contributions")
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test mode
        if len(sys.argv) > 2:
            run_id = int(sys.argv[2])
            test_hmarl_system(run_id=run_id, n_games=10)
        else:
            # Test all runs
            for run_id in [1, 2, 3]:
                try:
                    test_hmarl_system(run_id=run_id, n_games=5)
                except:
                    print(f"Run {run_id} not found")
    else:
        # Training mode
        results = run_hmarl_experiments(n_runs=3, timesteps_per_run=10_000_000)