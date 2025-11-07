"""
Simple Chess Trainer - Using reward shaping and PPO and using valid moves from env.

"""
import numpy as np
import gymnasium as gym
from pettingzoo.classic import chess_v6
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import os
import chess

class SimpleChessEnv(gym.Env):
    """Simple chess environment with automatic legal move handling"""
    
    def __init__(self):
        self.env = chess_v6.env(render_mode=None)
        # self.env = chess_v6.env(render_mode="human") # render in UI, use for testing only.
        self.env.reset()
        
        board_size = 8 * 8 * 111
        action_mask_size = 4672
        
        self.observation_space = gym.spaces.Box(
            low=0, high=1,
            shape=(board_size + action_mask_size,),
            dtype=np.float32
        )
        self.action_space = self.env.action_space('player_0')
        
        # Track episodes manually
        self.episode_reward = 0
        self.episode_length = 0
    
    def reset(self, **kwargs):
        self.env.reset()
        self.episode_reward = 0
        self.episode_length = 0
        obs, _, _, _, _ = self.env.last()
        return self._get_obs(obs), {}
    
    def step(self, action):
        # Get legal actions and ensure we only make legal moves
        current_obs = self.env.observe(self.env.agent_selection)
        
        if current_obs and 'action_mask' in current_obs:
            action_mask = current_obs['action_mask']
            legal_actions = np.where(action_mask)[0]
            
            # Map agent's action to legal action
            if len(legal_actions) > 0:
                if action >= len(action_mask) or not action_mask[action]:
                    action = legal_actions[action % len(legal_actions)]
        
        # Store board state before move for material calculation
        board_before = self.env.unwrapped.board.copy()
        material_before = self._calculate_material(board_before)
        
        # Execute move
        self.env.step(action)
        obs, reward, terminated, truncated, info = self.env.last()
        
        # Scale up game outcome rewards to keep them dominant
        if terminated:
            if reward == 1:  # Win
                reward = 10.0  # Big win bonus
            elif reward == -1:  # Loss  
                reward = -10.0  # Big loss penalty
            # Draw stays 0
        else:
            # Add material-based rewards (smaller scale)
            board_after = self.env.unwrapped.board
            material_after = self._calculate_material(board_after)
            
            # Reward material gain (when we capture opponent's pieces)
            material_diff = material_after - material_before
            reward += material_diff * 0.1  # Material bonuses: 0.1 to 0.9
            
            # Reward for giving check
            if board_after.is_check():
                reward += 0.2
        
        self.episode_reward += reward
        self.episode_length += 1
        
        # Add episode info when done
        if terminated or truncated:
            info['episode'] = {
                'r': self.episode_reward,
                'l': self.episode_length
            }
        
        return self._get_obs(obs), reward, terminated, truncated, info
    
    def _get_obs(self, obs):
        """Combine board observation with action mask"""
        if obs is None or 'observation' not in obs:
            board_obs = np.zeros((8*8*111,), dtype=np.float32)
            action_mask = np.ones(4672, dtype=np.float32)
            return np.concatenate([board_obs, action_mask])
        
        board_obs = obs['observation'].flatten().astype(np.float32)
        action_mask = obs['action_mask'].astype(np.float32)
        
        return np.concatenate([board_obs, action_mask])
    
    def _calculate_material(self, board):
        """Calculate material balance from white's perspective"""
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0  # King has no material value
        }
        
        white_material = 0
        black_material = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value
        
        # Return material difference (positive = white advantage)
        return white_material - black_material
    
    def close(self):
        self.env.close()

class AggressiveChessCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.wins = 0
        self.draws = 0
        self.losses = 0

        
    def _on_step(self) -> bool:
        if self.locals.get('dones') is not None:
            done_indices = [i for i, done in enumerate(self.locals['dones']) if done]
            
            if done_indices:
                infos = self.locals.get("infos", [])
                
                for i in done_indices:
                    if i < len(infos) and 'episode' in infos[i]:
                        self.episode_count += 1
                        
                        episode_reward = infos[i]['episode']['r']
                        episode_length = infos[i]['episode']['l']
                        
                        self.episode_rewards.append(episode_reward)
                        self.episode_lengths.append(episode_length)
                        
                        if episode_reward >= 8:
                            self.wins += 1
                        elif episode_reward <= -8:
                            self.losses += 1
                        else:
                            self.draws += 1
                        
                        if self.episode_count % 10 == 0:
                            win_rate = self.wins / self.episode_count * 100
                            print(f"Episode {self.episode_count}: Win {win_rate:.1f}%")
        
        return True

def train_aggressive_chess(timesteps=200000, run_id=1):
    print(f"Training Run {run_id} - {timesteps:,} timesteps")
    
    vec_env = DummyVecEnv([lambda: SimpleChessEnv() for _ in range(4)])
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=1e-3,
        n_steps=1024,
        batch_size=128,
        n_epochs=10,
        gamma=0.95,
        gae_lambda=0.9,
        clip_range=0.3,
        ent_coef=0.05,
        vf_coef=1.0,
        max_grad_norm=1.0,
        device="cpu",
        tensorboard_log=f"./logs/run_{run_id}/"
    )
    
    callback = AggressiveChessCallback(verbose=1)
    model.learn(total_timesteps=timesteps, callback=callback)
    
    os.makedirs("./models", exist_ok=True)
    model.save(f"./models/single_agent_{run_id}")
    vec_env.close()
    
    return model, callback

def test_aggressive_chess(model_path):
    """Test the aggressive chess agent"""
    
    print("🎮 Testing Aggressive Chess Agent...")
    
    model = PPO.load(model_path)
    env = SimpleChessEnv()
    
    obs, _ = env.reset()
    
    for step in range(250):  # Max moves
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 10 == 0:
            print(f"Step {step+1}: Reward {reward:.2f}")
        
        if terminated or truncated:
            if reward >= 8:
                print(f"🏆 WON after {step+1} steps! (Reward: {reward:.2f})")
            elif reward <= -8:
                print(f"💀 LOST after {step+1} steps (Reward: {reward:.2f})")
            else:
                print(f"🤝 DRAW after {step+1} steps (Reward: {reward:.2f})")
            break
    
    env.close()

def run_multiple_experiments(n_runs=3, timesteps_per_run=10_000_000):
    print(f"Running {n_runs} experiments - {timesteps_per_run:,} timesteps each")
    
    all_results = []
    
    for run_id in range(10, n_runs + 1):
        print(f"Starting run {run_id}/{n_runs}")
        
        try:
            model, callback = train_aggressive_chess(timesteps=timesteps_per_run, run_id=run_id)
            
            print(f"Run {run_id} finished - Episodes: {callback.episode_count}")
            
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
                    'avg_reward': np.mean(callback.episode_rewards),
                    'avg_length': np.mean(callback.episode_lengths)
                }
                
                all_results.append(results)
                
                print(f"Run {run_id} completed - Win: {results['win_rate']:.1f}%, Draw: {results['draw_rate']:.1f}%, Loss: {results['loss_rate']:.1f}%")
            else:
                print(f"Run {run_id} failed")
            
        except Exception as e:
            print(f"Run {run_id} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if all_results:
        win_rates = [r['win_rate'] for r in all_results]
        draw_rates = [r['draw_rate'] for r in all_results]
        avg_rewards = [r['avg_reward'] for r in all_results]
        
        print(f"Final: Win {np.mean(win_rates):.1f}%, Draw {np.mean(draw_rates):.1f}%, Reward {np.mean(avg_rewards):.2f}")
        import json
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"./models/chess_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                'experiment_config': {
                    'n_runs': n_runs,
                    'timesteps_per_run': timesteps_per_run,
                    'total_timesteps': n_runs * timesteps_per_run
                },
                'results': all_results,
                'summary': {
                    'mean_win_rate': np.mean(win_rates),
                    'std_win_rate': np.std(win_rates),
                    'mean_draw_rate': np.mean(draw_rates),
                    'std_draw_rate': np.std(draw_rates),
                    'mean_avg_reward': np.mean(avg_rewards),
                    'std_avg_reward': np.std(avg_rewards)
                }
            }, f, indent=2)
        
        print(f"Results saved: {results_file}")
    
    return all_results

if __name__ == "__main__":
    run_multiple_experiments(n_runs=13, timesteps_per_run=100_000)
 

    # test_aggressive_chess(r"models/single_agent_1.zip")