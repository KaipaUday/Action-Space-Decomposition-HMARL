"""
HMARL Chess Environment with Sequential Agent Handoff
Opening Agent (1-15) → Middle Agent (16-40) → Endgame Agent (40+)
"""
import numpy as np
import gymnasium as gym
from pettingzoo.classic import chess_v6
import chess

class HMARLChessEnv(gym.Env):
    """HMARL Chess Environment with sequential agent handoff"""
    
    def __init__(self):
        self.env = chess_v6.env(render_mode=None)
        # self.env = chess_v6.env(render_mode="human")
        self.env.reset()
        
        board_size = 8 * 8 * 111
        action_mask_size = 4672
        
        self.observation_space = gym.spaces.Box(
            low=0, high=1,
            shape=(board_size + action_mask_size,),
            dtype=np.float32
        )
        self.action_space = self.env.action_space('player_0')
        
        # Game state
        self.move_count = 0
        self.game_reward = 0
        self.game_length = 0
        
        # Agent tracking
        self.current_agent = "opening"  # opening, middle, endgame
        self.agent_rewards = {"opening": 0, "middle": 0, "endgame": 0}
        self.agent_moves = {"opening": 0, "middle": 0, "endgame": 0}
        
        # Episode tracking for each agent
        self.episode_data = {
            "opening": {"obs": [], "actions": [], "rewards": [], "dones": []},
            "middle": {"obs": [], "actions": [], "rewards": [], "dones": []},
            "endgame": {"obs": [], "actions": [], "rewards": [], "dones": []}
        }
    
    def reset(self, **kwargs):
        self.env.reset()
        self.move_count = 0
        self.game_reward = 0
        self.game_length = 0
        
        # Reset agent tracking
        self.current_agent = "opening"
        self.agent_rewards = {"opening": 0, "middle": 0, "endgame": 0}
        self.agent_moves = {"opening": 0, "middle": 0, "endgame": 0}
        
        # Clear episode data
        for agent in self.episode_data:
            self.episode_data[agent] = {"obs": [], "actions": [], "rewards": [], "dones": []}
        
        obs, _, _, _, _ = self.env.last()
        return self._get_obs(obs), {}
    
    def step(self, action):
        # Store observation and action for current agent
        current_obs = self.env.observe(self.env.agent_selection)
        processed_obs = self._get_obs(current_obs)
        
        # Map illegal actions to legal ones
        if current_obs and 'action_mask' in current_obs:
            action_mask = current_obs['action_mask']
            legal_actions = np.where(action_mask)[0]
            
            if len(legal_actions) > 0:
                if action >= len(action_mask) or not action_mask[action]:
                    action = legal_actions[action % len(legal_actions)]
        
        # Store data for current agent
        self.episode_data[self.current_agent]["obs"].append(processed_obs)
        self.episode_data[self.current_agent]["actions"].append(action)
        
        # Calculate material before move
        board_before = self.env.unwrapped.board.copy()
        material_before = self._calculate_material(board_before)
        
        # Execute move
        self.env.step(action)
        obs, reward, terminated, truncated, info = self.env.last()
        
        self.move_count += 1
        self.game_length += 1
        
        # Calculate agent-specific reward
        agent_reward = 0
        
        if terminated:
            # Game ended - distribute final reward
            if reward == 1:
                final_reward = 10.0
            elif reward == -1:
                final_reward = -10.0
            else:
                final_reward = 0
            
            # Give final reward to current agent
            agent_reward = final_reward
            self.game_reward = final_reward
            
        else:
            # Calculate incremental rewards
            board_after = self.env.unwrapped.board
            material_after = self._calculate_material(board_after)
            material_diff = material_after - material_before
            
            # Base material reward
            agent_reward = material_diff * 0.1
            
            # Agent-specific bonuses
            if self.current_agent == "opening":
                agent_reward += self._opening_rewards(board_before, board_after)
            elif self.current_agent == "middle":
                agent_reward += self._middle_rewards(board_before, board_after)
            elif self.current_agent == "endgame":
                agent_reward += self._endgame_rewards(board_before, board_after)
        
        # Update agent tracking
        self.agent_rewards[self.current_agent] += agent_reward
        self.agent_moves[self.current_agent] += 1
        
        # Store reward and done status for current agent
        self.episode_data[self.current_agent]["rewards"].append(agent_reward)
        self.episode_data[self.current_agent]["dones"].append(terminated or truncated)
        
        # Check for agent handoff
        next_agent = self._get_next_agent()
        if next_agent != self.current_agent and not (terminated or truncated):
            # Agent handoff - current agent's episode ends, next agent starts
            self.current_agent = next_agent
        
        # Prepare info for episode tracking
        if terminated or truncated:
            info['episode_data'] = {
                'agents': self.agent_rewards,
                'moves': self.agent_moves,
                'total_reward': self.game_reward,
                'total_length': self.game_length
            }
        
        return self._get_obs(obs), agent_reward, terminated, truncated, info
    
    def _get_next_agent(self):
        """Determine which agent should act based on move count"""
        if self.move_count <= 15:
            return "opening"
        elif self.move_count <= 40:
            return "middle"
        else:
            return "endgame"
    
    def _opening_rewards(self, board_before, board_after):
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
    
    def _middle_rewards(self, board_before, board_after):
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
    
    def _endgame_rewards(self, board_before, board_after):
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
    
    def _calculate_material(self, board):
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
        if obs is None or 'observation' not in obs:
            board_obs = np.zeros((8*8*111,), dtype=np.float32)
            action_mask = np.ones(4672, dtype=np.float32)
            return np.concatenate([board_obs, action_mask])
        
        board_obs = obs['observation'].flatten().astype(np.float32)
        action_mask = obs['action_mask'].astype(np.float32)
        
        return np.concatenate([board_obs, action_mask])
    
    def get_episode_data(self, agent_type):
        """Get episode data for specific agent"""
        return self.episode_data[agent_type]
    
    def close(self):
        self.env.close()