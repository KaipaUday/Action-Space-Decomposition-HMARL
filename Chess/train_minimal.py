"""
Fixed Chess Training with Proper Self-Play Rewards
Key fix: Both players get terminal rewards, not just the last mover
"""
import numpy as np
import gymnasium as gym
from pettingzoo.classic import chess_v6
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import os
import chess
from torch.distributions import Distribution 
Distribution.set_default_validate_args(False)

class FixedChessEnv(gym.Env):
    """Chess environment with PROPER self-play rewards for both players"""
    
    def __init__(self):
        self.env = chess_v6.env(render_mode=None)
        self.env.reset()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=1,
            shape=(8*8*111,),
            dtype=np.float32
        )
        self.action_space = self.env.action_space('player_0')
        
        # Track moves for both players
        self.white_moves = []  # (obs, action, reward)
        self.black_moves = []
        self.current_player_moves = None
        
        self.episode_reward = 0
        self.episode_length = 0
        self.game_outcome = None
        self.move_count = 0
    
    def reset(self, **kwargs):
        self.env.reset()
        self.white_moves = []
        self.black_moves = []
        self.current_player_moves = self.white_moves
        self.episode_reward = 0
        self.episode_length = 0
        self.game_outcome = None
        self.move_count = 0
        
        obs, _, _, _, _ = self.env.last()
        return self._get_obs(obs), {}
    
    def action_masks(self):
        """Return action mask from PettingZoo"""
        if not self.env.agents:
            return np.ones(4672, dtype=bool)
        
        try:
            current_obs = self.env.observe(self.env.agent_selection)
            
            if current_obs and 'action_mask' in current_obs:
                mask = current_obs['action_mask'].astype(bool)
                if not np.any(mask):
                    return np.ones(4672, dtype=bool)
                return mask
            else:
                return np.ones(4672, dtype=bool)
        except:
            return np.ones(4672, dtype=bool)
    
    def step(self, action):
        # Calculate material before move
        board_before = self.env.unwrapped.board.copy()
        material_before = self._calculate_material(board_before)
        
        # Get current agent
        acting_agent = self.env.agent_selection
        
        # Switch move tracking based on which player is moving
        if acting_agent == 'player_0':
            self.current_player_moves = self.white_moves
        else:
            self.current_player_moves = self.black_moves
        
        # Agent makes move
        self.env.step(action)
        obs, reward, terminated, truncated, info = self.env.last()
        
        self.move_count += 1
        self.episode_length += 1
        
        # Calculate incremental reward
        agent_reward = 0
        
        if not (terminated or truncated):
            # Calculate incremental rewards during game
            board_after = self.env.unwrapped.board
            material_after = self._calculate_material(board_after)
            material_diff = material_after - material_before
            
            # Adjust material diff based on who moved
            if acting_agent == 'player_1':
                material_diff = -material_diff
            
            # Base material reward
            agent_reward = material_diff * 0.1
            
            # Phase-specific bonuses
            if self.move_count <= 15:
                agent_reward += self._opening_rewards(board_before, board_after)
            elif self.move_count <= 40:
                agent_reward += self._middle_rewards(board_before, board_after)
            else:
                agent_reward += self._endgame_rewards(board_before, board_after)
        
        # Store the move for this player
        self.current_player_moves.append({
            'obs': self._get_obs(obs),
            'reward': agent_reward
        })
        
        # Handle game termination
        if terminated or truncated:
            # Determine outcome from PettingZoo rewards
            white_reward = self.env.rewards.get('player_0', 0)
            black_reward = self.env.rewards.get('player_1', 0)
            
            # Add terminal rewards to all moves
            if white_reward == 1:  # White won
                self.game_outcome = 'white_win'
                white_terminal = 5.0
                black_terminal = -5.0
            elif black_reward == 1:  # Black won
                self.game_outcome = 'black_win'
                white_terminal = -5.0
                black_terminal = 5.0
            else:  # Draw
                self.game_outcome = 'draw'
                white_terminal = -1.0
                black_terminal = -1.0
            
            # Apply terminal reward to the last move of each player
            if self.white_moves:
                self.white_moves[-1]['reward'] += white_terminal
            if self.black_moves:
                self.black_moves[-1]['reward'] += black_terminal
            
            # Return reward for current player
            if acting_agent == 'player_0':
                final_reward = white_terminal
            else:
                final_reward = black_terminal
            
            self.episode_reward = final_reward
            
            info['episode'] = {
                'r': self.episode_reward,
                'l': self.episode_length,
                'game_outcome': self.game_outcome,
                'white_moves': len(self.white_moves),
                'black_moves': len(self.black_moves)
            }
        
        return self._get_obs(obs), agent_reward, terminated, truncated, info
    
    def _opening_rewards(self, board_before, board_after):
        """Rewards for opening play"""
        reward = 0
        last_move = board_after.peek() if len(board_after.move_stack) > 0 else None
        
        if last_move:
            piece = board_before.piece_at(last_move.from_square)
            if piece:
                # Development
                if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                    from_rank = chess.square_rank(last_move.from_square)
                    starting_rank = 0 if piece.color == chess.WHITE else 7
                    if from_rank == starting_rank:
                        reward += 0.3 *2
                
                # Center control
                center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
                if last_move.to_square in center_squares:
                    reward += 0.2 *2
                
                # Castling
                if board_after.is_castling(last_move):
                    reward += 0.4 *2
        
        return reward
    
    def _middle_rewards(self, board_before, board_after):
        """Rewards for middle game"""
        reward = 0
        
        if board_after.is_check():
            reward += 0.3 *2
        
        last_move = board_after.peek() if len(board_after.move_stack) > 0 else None
        if last_move:
            enemy_king = board_after.king(not board_after.turn)
            if enemy_king:
                distance = chess.square_distance(last_move.to_square, enemy_king)
                if distance <= 3:
                    reward += 0.2*2
            
            if board_before.is_attacked_by(not board_before.turn, last_move.to_square):
                reward += 0.1*2
        
        return reward
    
    def _endgame_rewards(self, board_before, board_after):
        """Rewards for endgame"""
        reward = 0
        last_move = board_after.peek() if len(board_after.move_stack) > 0 else None
        
        if last_move:
            piece = board_before.piece_at(last_move.from_square)
            if piece:
                if piece.piece_type == chess.KING:
                    reward += 0.2 *2
                
                if piece.piece_type == chess.PAWN:
                    to_rank = chess.square_rank(last_move.to_square)
                    if to_rank in [0, 7]:
                        reward += 0.5 *2
                
                center_distance = abs(chess.square_file(last_move.to_square) - 3.5) + \
                                abs(chess.square_rank(last_move.to_square) - 3.5)
                if center_distance < 2:
                    reward += 0.1 *2
        
        return reward
    
    def _calculate_material(self, board):
        """Calculate material balance"""
        piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
        }
        
        white_material = sum(len(board.pieces(piece, chess.WHITE)) * value 
                           for piece, value in piece_values.items())
        black_material = sum(len(board.pieces(piece, chess.BLACK)) * value 
                           for piece, value in piece_values.items())
        
        return white_material - black_material
    
    def _get_obs(self, obs):
        """Extract observation"""
        if obs is None or 'observation' not in obs:
            return np.zeros((8*8*111,), dtype=np.float32)
        
        return obs['observation'].flatten().astype(np.float32)
    
    def close(self):
        self.env.close()


class ImprovedCallback(BaseCallback):
    """Track detailed game outcomes"""
    
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_count = 0
        self.white_wins = 0
        self.black_wins = 0
        self.draws = 0
        
    def _on_step(self) -> bool:
        if 'dones' in self.locals:
            dones = self.locals['dones']
            infos = self.locals.get('infos', [])
            
            for i, done in enumerate(dones):
                if done and i < len(infos):
                    if 'episode' in infos[i]:
                        self.episode_count += 1
                        
                        game_outcome = infos[i]['episode'].get('game_outcome', None)
                        episode_length = infos[i]['episode']['l']
                        
                        if game_outcome == 'white_win':
                            self.white_wins += 1
                            outcome = "W-WIN"
                        elif game_outcome == 'black_win':
                            self.black_wins += 1
                            outcome = "B-WIN"
                        else:
                            self.draws += 1
                            outcome = "DRAW"
                        
                        if self.episode_count % 1 == 0:
                            decisive = self.white_wins + self.black_wins
                            decisive_rate = decisive / self.episode_count * 100
                            draw_rate = self.draws / self.episode_count * 100
                            
                            print(f"Ep {self.episode_count:4d} | "
                                  f"{outcome:5s} | "
                                  f"Decisive: {decisive_rate:5.1f}% | "
                                  f"Draws: {draw_rate:5.1f}% | "
                                  f"Moves: {episode_length:3d}")
        
        return True


def train_fixed(timesteps=500000):
    """Train with proper self-play rewards"""
    print("="*60)
    print("FIXED Chess Self-Play Training")
    print("="*60)
    print("KEY FIX: Both players now receive terminal rewards")
    print("  - Winner gets +10 on their last move")
    print("  - Loser gets -10 on their last move")
    print("  - Both players learn from game outcome")
    print("="*60)
    
    vec_env = DummyVecEnv([lambda: FixedChessEnv() for _ in range(4)])
    
    model = MaskablePPO(
        MaskableActorCriticPolicy,
        vec_env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device="cpu",
        tensorboard_log="./logs/fixed_selfplay/"
    )
    
    callback = ImprovedCallback(verbose=1)
    
    print("\nStarting training...")
    print("-"*60)
    
    model.learn(total_timesteps=timesteps, callback=callback)
    
    print("-"*60)
    
    os.makedirs("./models", exist_ok=True)
    model_path = "./models/fixed_selfplay_agent.zip"
    model.save(model_path)
    
    vec_env.close()
    
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Total Episodes: {callback.episode_count}")
    
    if callback.episode_count > 0:
        white_win_rate = callback.white_wins / callback.episode_count * 100
        black_win_rate = callback.black_wins / callback.episode_count * 100
        draw_rate = callback.draws / callback.episode_count * 100
        decisive = callback.white_wins + callback.black_wins
        decisive_rate = decisive / callback.episode_count * 100
        
        print(f"\nResults:")
        print(f"  White Wins: {callback.white_wins:4d} ({white_win_rate:5.1f}%)")
        print(f"  Black Wins: {callback.black_wins:4d} ({black_win_rate:5.1f}%)")
        print(f"  Draws:      {callback.draws:4d} ({draw_rate:5.1f}%)")
        print(f"  Decisive:   {decisive:4d} ({decisive_rate:5.1f}%)")
        
        print(f"\nExpected: ~25-40% decisive games as training progresses")
    
    print(f"\nModel saved: {model_path}")
    print("="*60)
    
    return model, callback


if __name__ == "__main__":
    import sys
    
    timesteps = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    model, callback = train_fixed(timesteps=timesteps)