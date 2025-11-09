"""
Simultaneous MARL Chess Training
All three agents train on the same games with proper handoffs
- Opening agent: moves 1-15
- Middle agent: moves 16-40  
- Endgame agent: moves 40+
"""
import numpy as np
from pettingzoo.classic import chess_v6
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks
import os
import chess
from collections import deque
from torch.distributions import Distribution 
Distribution.set_default_validate_args(False)

class ExperienceBuffer:
    """Store experiences for each agent"""
    def __init__(self, max_size=10000):
        self.observations = deque(maxlen=max_size)
        self.actions = deque(maxlen=max_size)
        self.rewards = deque(maxlen=max_size)
        self.dones = deque(maxlen=max_size)
        self.action_masks = deque(maxlen=max_size)
        self.next_observations = deque(maxlen=max_size)
        
    def add(self, obs, action, reward, done, action_mask, next_obs):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.action_masks.append(action_mask)
        self.next_observations.append(next_obs)
    
    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.action_masks.clear()
        self.next_observations.clear()
    
    def size(self):
        return len(self.observations)


def calculate_material(board):
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


def opening_rewards(board_before, board_after):
    """Rewards for opening play (moves 1-15)"""
    reward = 0
    last_move = board_after.peek() if len(board_after.move_stack) > 0 else None
    
    if last_move:
        piece = board_before.piece_at(last_move.from_square)
        if piece:
            # Reward piece development
            if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                from_rank = chess.square_rank(last_move.from_square)
                starting_rank = 0 if piece.color == chess.WHITE else 7
                if from_rank == starting_rank:
                    reward += 0.3
            
            # Reward center control
            center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
            if last_move.to_square in center_squares:
                reward += 0.2
            
            # Reward castling
            if board_after.is_castling(last_move):
                reward += 0.4
    
    return reward


def middle_rewards(board_before, board_after):
    """Rewards for middle game (moves 16-40)"""
    reward = 0
    
    # Reward tactical play
    if board_after.is_check():
        reward += 0.3
    
    # Reward piece activity
    last_move = board_after.peek() if len(board_after.move_stack) > 0 else None
    if last_move:
        # Reward moving toward enemy king
        enemy_king = board_after.king(not board_after.turn)
        if enemy_king:
            distance = chess.square_distance(last_move.to_square, enemy_king)
            if distance <= 3:
                reward += 0.2
        
        # Reward attacking moves
        if board_before.is_attacked_by(not board_before.turn, last_move.to_square):
            reward += 0.1
    
    return reward


def endgame_rewards(board_before, board_after):
    """Rewards for endgame (moves 40+)"""
    reward = 0
    last_move = board_after.peek() if len(board_after.move_stack) > 0 else None
    
    if last_move:
        piece = board_before.piece_at(last_move.from_square)
        if piece:
            # Reward king activity
            if piece.piece_type == chess.KING:
                reward += 0.2
            
            # Reward pawn promotion
            if piece.piece_type == chess.PAWN:
                to_rank = chess.square_rank(last_move.to_square)
                if to_rank in [0, 7]:
                    reward += 0.5
            
            # Reward centralization
            center_distance = abs(chess.square_file(last_move.to_square) - 3.5) + \
                            abs(chess.square_rank(last_move.to_square) - 3.5)
            if center_distance < 2:
                reward += 0.1
    
    return reward


def play_game_with_agents(opening_model, middle_model, endgame_model, 
                          opening_buffer, middle_buffer, endgame_buffer):
    """
    Play one complete game with all three agents
    Each agent controls their phase and stores experiences
    """
    env = chess_v6.env(render_mode=None)
    env.reset()
    
    move_count = 0
    game_done = False
    
    # Track statistics
    phase_moves = {"opening": 0, "middle": 0, "endgame": 0}
    phase_rewards = {"opening": 0, "middle": 0, "endgame": 0}
    
    while not game_done and env.agents:
        # Determine current phase
        if move_count <= 15:
            current_phase = "opening"
            current_model = opening_model
            current_buffer = opening_buffer
            reward_fn = opening_rewards
        elif move_count <= 40:
            current_phase = "middle"
            current_model = middle_model
            current_buffer = middle_buffer
            reward_fn = middle_rewards
        else:
            current_phase = "endgame"
            current_model = endgame_model
            current_buffer = endgame_buffer
            reward_fn = endgame_rewards
        
        # Get observation
        obs, reward, termination, truncation, info = env.last()
        
        if termination or truncation:
            env.step(None)
            continue
        
        if not obs or 'observation' not in obs:
            break
        
        # Get action from current phase agent
        obs_flat = obs['observation'].flatten().astype(np.float32)
        action_mask = obs['action_mask'].astype(bool)
        
        # Get action from model
        action, _ = current_model.predict(obs_flat, action_masks=action_mask, deterministic=False)
        
        # Calculate material before move
        board_before = env.unwrapped.board.copy()
        material_before = calculate_material(board_before)
        
        # Get acting agent
        acting_agent = env.agent_selection
        
        # Take action
        env.step(action)
        next_obs, reward, termination, truncation, info = env.last()
        
        # Calculate reward
        step_reward = 0
        
        if termination or truncation:
            # Game ended
            acting_agent_reward = env.rewards.get(acting_agent, 0)
            
            if acting_agent_reward == 1:
                step_reward = 10.0
            elif acting_agent_reward == -1:
                step_reward = -10.0
            else:
                step_reward = 0
            
            game_done = True
        else:
            # Calculate incremental reward
            board_after = env.unwrapped.board
            material_after = calculate_material(board_after)
            material_diff = material_after - material_before
            
            # Adjust for black's moves
            if acting_agent == 'player_1':
                material_diff = -material_diff
            
            # Base material reward
            step_reward = material_diff * 0.1
            
            # Phase-specific bonus
            step_reward += reward_fn(board_before, board_after)
        
        # Store experience in current phase buffer
        next_obs_flat = next_obs['observation'].flatten().astype(np.float32) if next_obs and 'observation' in next_obs else obs_flat
        next_action_mask = next_obs['action_mask'].astype(bool) if next_obs and 'action_mask' in next_obs else action_mask
        
        current_buffer.add(
            obs_flat,
            action,
            step_reward,
            termination or truncation,
            action_mask,
            next_obs_flat
        )
        
        # Track statistics
        phase_moves[current_phase] += 1
        phase_rewards[current_phase] += step_reward
        
        # Count moves (only player_0's moves)
        if acting_agent == 'player_0':
            move_count += 1
    
    env.close()
    
    return phase_moves, phase_rewards, game_done


def train_marl_simultaneous(n_games=1000, update_freq=10):
    """
    Train all three MARL agents simultaneously
    
    Args:
        n_games: Number of games to play
        update_freq: Update models every N games
    """
    print("="*60)
    print("Simultaneous MARL Chess Training")
    print("="*60)
    print(f"Games: {n_games}")
    print(f"Update frequency: every {update_freq} games")
    print(f"All agents train on same games with proper handoffs")
    print("="*60)
    
    # Create models
    print("\nInitializing models...")
    
    # Dummy observation and action spaces
    obs_shape = (8*8*111,)
    action_space_size = 4672
    
    from gymnasium import spaces
    dummy_obs_space = spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32)
    dummy_action_space = spaces.Discrete(action_space_size)
    
    # Create models (we'll use them for prediction, custom training loop for updates)
    opening_model = MaskablePPO(
        MaskableActorCriticPolicy,
        env=None,  # We'll handle environment manually
        verbose=0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        device="cpu"
    )
    
    middle_model = MaskablePPO(
        MaskableActorCriticPolicy,
        env=None,
        verbose=0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        device="cpu"
    )
    
    endgame_model = MaskablePPO(
        MaskableActorCriticPolicy,
        env=None,
        verbose=0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        device="cpu"
    )
    
    # Initialize models with dummy environment
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    class DummyEnv:
        def __init__(self):
            self.observation_space = dummy_obs_space
            self.action_space = dummy_action_space
        def reset(self):
            return np.zeros(obs_shape, dtype=np.float32), {}
        def step(self, action):
            return np.zeros(obs_shape, dtype=np.float32), 0, False, False, {}
        def action_masks(self):
            return np.ones(action_space_size, dtype=bool)
    
    dummy_vec_env = DummyVecEnv([DummyEnv for _ in range(1)])
    
    opening_model.set_env(dummy_vec_env)
    middle_model.set_env(dummy_vec_env)
    endgame_model.set_env(dummy_vec_env)
    
    # Create experience buffers
    opening_buffer = ExperienceBuffer()
    middle_buffer = ExperienceBuffer()
    endgame_buffer = ExperienceBuffer()
    
    # Training statistics
    total_games = 0
    wins = 0
    draws = 0
    losses = 0
    
    print("\nStarting training...")
    print("-"*60)
    
    for game in range(1, n_games + 1):
        # Play one game with all agents
        phase_moves, phase_rewards, game_done = play_game_with_agents(
            opening_model, middle_model, endgame_model,
            opening_buffer, middle_buffer, endgame_buffer
        )
        
        total_games += 1
        
        # Track outcomes (based on total reward)
        total_reward = sum(phase_rewards.values())
        if total_reward > 5:
            wins += 1
            outcome = "WIN"
        elif total_reward < -5:
            losses += 1
            outcome = "LOSS"
        else:
            draws += 1
            outcome = "DRAW"
        
        # Print progress
        if game % 10 == 0:
            win_rate = wins / total_games * 100
            print(f"Game {game:4d} | {outcome:4s} | "
                  f"W:{wins:3d} D:{draws:3d} L:{losses:3d} | "
                  f"WR:{win_rate:5.1f}% | "
                  f"Buffers: O:{opening_buffer.size()} M:{middle_buffer.size()} E:{endgame_buffer.size()}")
        
        # Update models periodically
        if game % update_freq == 0 and game > 0:
            print(f"\n  Updating models at game {game}...")
            
            # Note: Actual model updates would require implementing custom PPO update
            # For now, we just clear buffers (simplified)
            # In a full implementation, you'd call model.train() with the buffer data
            
            if opening_buffer.size() > 0:
                print(f"    Opening: {opening_buffer.size()} experiences")
                # opening_model.train_on_buffer(opening_buffer)  # Custom implementation needed
                opening_buffer.clear()
            
            if middle_buffer.size() > 0:
                print(f"    Middle: {middle_buffer.size()} experiences")
                # middle_model.train_on_buffer(middle_buffer)  # Custom implementation needed
                middle_buffer.clear()
            
            if endgame_buffer.size() > 0:
                print(f"    Endgame: {endgame_buffer.size()} experiences")
                # endgame_model.train_on_buffer(endgame_buffer)  # Custom implementation needed
                endgame_buffer.clear()
            
            print()
    
    print("-"*60)
    
    # Save models
    os.makedirs("./models", exist_ok=True)
    opening_model.save("./models/marl_sim_opening.zip")
    middle_model.save("./models/marl_sim_middle.zip")
    endgame_model.save("./models/marl_sim_endgame.zip")
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Total Games: {total_games}")
    print(f"\nResults:")
    print(f"  Wins:   {wins:4d} ({wins/total_games*100:5.1f}%)")
    print(f"  Draws:  {draws:4d} ({draws/total_games*100:5.1f}%)")
    print(f"  Losses: {losses:4d} ({losses/total_games*100:5.1f}%)")
    print(f"\nModels saved:")
    print(f"  ./models/marl_sim_opening.zip")
    print(f"  ./models/marl_sim_middle.zip")
    print(f"  ./models/marl_sim_endgame.zip")
    print("="*60)
    
    return opening_model, middle_model, endgame_model


if __name__ == "__main__":
    print("\n" + "="*60)
    print("EXPERIMENT: Simultaneous MARL Chess Training")
    print("="*60)
    print("\nAll three agents train on the same games:")
    print("  - Opening agent controls moves 1-15")
    print("  - Middle agent controls moves 16-40")
    print("  - Endgame agent controls moves 40+")
    print("\nEach agent stores experiences and updates periodically")
    print("="*60)
    print("\nNOTE: This is a simplified implementation.")
    print("Full PPO updates require custom training loop.")
    print("For production, consider using RLlib or custom PPO implementation.")
    print("="*60)
    
    try:
        opening, middle, endgame = train_marl_simultaneous(n_games=100, update_freq=10)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
