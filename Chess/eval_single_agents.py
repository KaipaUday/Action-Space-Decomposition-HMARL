"""
Evaluate Single-Agent Models: 10M vs 100K timesteps
Compare agents trained for different durations
"""
import numpy as np
from pettingzoo.classic import chess_v6
from stable_baselines3 import PPO
import json
from datetime import datetime
import os


class SingleAgentEvaluator:
    def __init__(self):
        self.results = {
            'matches': [],
            'summary': {}
        }
    
    def play_match(self, white_model, black_model, white_name, black_name, match_id):
        """Play a single chess match between two agents"""
        env = chess_v6.env(render_mode=None)
        env.reset()
        
        move_count = 0
        max_moves = 250
        
        # Record moves in PGN format
        move_history = []
        
        # Track material over time
        material_history = []
        
        while env.agents and move_count < max_moves:
            current_player = env.agent_selection
            obs, reward, terminated, truncated, info = env.last()
            
            if terminated or truncated:
                break
            
            # Get observation
            if obs is None or 'observation' not in obs:
                break
            
            board_obs = obs['observation'].flatten()
            action_mask = obs['action_mask']
            combined_obs = np.concatenate([board_obs, action_mask])
            
            # Select model based on current player
            model = white_model if current_player == 'player_0' else black_model
            
            # Get action (stochastic for variety)
            action, _ = model.predict(combined_obs, deterministic=False)
            
            # Ensure legal move
            legal_actions = np.where(action_mask)[0]
            if len(legal_actions) > 0:
                if action >= len(action_mask) or not action_mask[action]:
                    action = legal_actions[action % len(legal_actions)]
            
            env.step(action)
            move_count += 1
            
            # Record move in chess notation and material AFTER the move is made
            try:
                # Use the PettingZoo board directly
                pz_board = env.unwrapped.board
                if len(pz_board.move_stack) > 0:
                    last_move = pz_board.move_stack[-1]
                    # Convert to SAN notation
                    temp_board = pz_board.copy()
                    temp_board.pop()  # Remove last move to get position before
                    chess_move = temp_board.san(last_move)
                    move_history.append(chess_move)
                    
                    # Calculate material difference directly from PettingZoo board (White - Black)
                    material_diff = self._calculate_material(pz_board)
                    material_history.append(material_diff)
            except Exception as e:
                pass  # Skip if move recording fails
        
        # Determine result
        final_obs, final_reward, terminated, truncated, info = env.last()
        
        if terminated:
            if final_reward == 1:
                result = "white_win"
            elif final_reward == -1:
                result = "black_win"
            else:
                result = "draw"
        else:
            result = "draw"  # Max moves reached
        
        env.close()
        
        # Calculate final material difference
        final_material_diff = material_history[-1] if material_history else 0
        
        match_result = {
            'match_id': match_id,
            'white': white_name,
            'black': black_name,
            'result': result,
            'moves': move_count,
            'final_material_diff': final_material_diff
        }
        
        # Add move history for decisive games (wins)
        if result in ['white_win', 'black_win']:
            match_result['move_history'] = move_history
            match_result['material_history'] = material_history
            match_result['pgn'] = self._create_pgn(white_name, black_name, move_history, result)
        
        return match_result
    
    def _create_pgn(self, white_name, black_name, moves, result):
        """Create PGN format game notation"""
        pgn_result = "1-0" if result == "white_win" else "0-1"
        
        pgn = f'[White "{white_name}"]\n'
        pgn += f'[Black "{black_name}"]\n'
        pgn += f'[Result "{pgn_result}"]\n\n'
        
        # Format moves
        move_pairs = []
        for i in range(0, len(moves), 2):
            move_num = i // 2 + 1
            white_move = moves[i] if i < len(moves) else ""
            black_move = moves[i+1] if i+1 < len(moves) else ""
            if black_move:
                move_pairs.append(f"{move_num}. {white_move} {black_move}")
            else:
                move_pairs.append(f"{move_num}. {white_move}")
        
        pgn += " ".join(move_pairs)
        pgn += f" {pgn_result}"
        
        return pgn
    
    def _calculate_material(self, board):
        """Calculate material balance (White - Black)"""
        import chess
        
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
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
        
        return white_material - black_material
    
    def evaluate_agents(self, agents_10m, agents_100k, n_matches=1000):
        """
        Evaluate 10M timestep agents vs 100K timestep agents
        
        Args:
            agents_10m: List of paths to 10M timestep models (agents 1, 2, 3)
            agents_100k: List of paths to 100K timestep models (agents 10, 11, 12)
            n_matches: Number of matches per pairing
        """
        print(f"Loading {len(agents_10m)} agents trained for 10M timesteps...")
        models_10m = [PPO.load(path) for path in agents_10m]
        
        print(f"Loading {len(agents_100k)} agents trained for 100K timesteps...")
        models_100k = [PPO.load(path) for path in agents_100k]
        
        total_matches = len(models_10m) * len(models_100k) * n_matches * 2  # *2 for color swap
        match_count = 0
        
        print(f"\nStarting evaluation: {total_matches} total matches")
        print("=" * 60)
        
        # Play all combinations
        for i, model_10m in enumerate(models_10m):
            for j, model_100k in enumerate(models_100k):
                name_10m = f"10M_{i+1}"
                name_100k = f"100K_{j+10}"
                
                # Play n_matches with 10M agent as white
                for match in range(n_matches):
                    match_count += 1
                    result = self.play_match(
                        model_10m, model_100k,
                        name_10m, name_100k,
                        match_count
                    )
                    self.results['matches'].append(result)
                    
                    if match_count % 100 == 0:
                        print(f"Completed {match_count}/{total_matches} matches...")
                
                # Play n_matches with 100K agent as white (color swap)
                for match in range(n_matches):
                    match_count += 1
                    result = self.play_match(
                        model_100k, model_10m,
                        name_100k, name_10m,
                        match_count
                    )
                    self.results['matches'].append(result)
                    
                    if match_count % 100 == 0:
                        print(f"Completed {match_count}/{total_matches} matches...")
        
        print("=" * 60)
        print("Evaluation complete!")
        
        # Calculate summary statistics
        self._calculate_summary()
        self._print_results()
        
        return self.results
    
    def _calculate_summary(self):
        """Calculate win/loss/draw statistics"""
        wins_10m = 0
        wins_100k = 0
        draws = 0
        
        material_10m = []
        material_100k = []
        
        for match in self.results['matches']:
            material_diff = match.get('final_material_diff', 0)
            
            if match['result'] == 'white_win':
                if '10M' in match['white']:
                    wins_10m += 1
                    material_10m.append(material_diff)
                else:
                    wins_100k += 1
                    material_100k.append(-material_diff)
            elif match['result'] == 'black_win':
                if '10M' in match['black']:
                    wins_10m += 1
                    material_10m.append(-material_diff)
                else:
                    wins_100k += 1
                    material_100k.append(material_diff)
            else:
                draws += 1
        
        total = len(self.results['matches'])
        
        self.results['summary'] = {
            'total_matches': total,
            'wins_10m': wins_10m,
            'wins_100k': wins_100k,
            'draws': draws,
            'win_rate_10m': wins_10m / total * 100,
            'win_rate_100k': wins_100k / total * 100,
            'draw_rate': draws / total * 100,
            'avg_material_10m': np.mean(material_10m) if material_10m else 0,
            'avg_material_100k': np.mean(material_100k) if material_100k else 0
        }
    
    def _print_results(self):
        """Print formatted results"""
        summary = self.results['summary']
        
        print("\n" + "=" * 60)
        print("SINGLE-AGENT EVALUATION: 10M vs 100K Timesteps")
        print("=" * 60)
        print(f"Total Matches: {summary['total_matches']}")
        print(f"\n10M Timestep Agents:")
        print(f"  Wins: {summary['wins_10m']} ({summary['win_rate_10m']:.1f}%)")
        print(f"  Avg Material Advantage: {summary['avg_material_10m']:.1f}")
        print(f"\n100K Timestep Agents:")
        print(f"  Wins: {summary['wins_100k']} ({summary['win_rate_100k']:.1f}%)")
        print(f"  Avg Material Advantage: {summary['avg_material_100k']:.1f}")
        print(f"\nDraws: {summary['draws']} ({summary['draw_rate']:.1f}%)")
        print("=" * 60)
    
    def save_results(self, filename=None):
        """Save results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"single_agent_eval_{timestamp}.json"
        
        os.makedirs("results", exist_ok=True)
        filepath = os.path.join("results", filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to: {filepath}")
        
        # Save winning games separately
        self._save_winning_games()
        
        return filepath
    
    def _save_winning_games(self):
        """Save PGN files for all winning games"""
        os.makedirs("results/single_agent_games", exist_ok=True)
        
        winning_games = [m for m in self.results['matches'] if m['result'] in ['white_win', 'black_win']]
        
        print(f"\nSaving {len(winning_games)} winning games...")
        
        for match in winning_games:
            if 'pgn' in match:
                filename = f"match_{match['match_id']}_{match['white']}_vs_{match['black']}_{match['result']}.pgn"
                filepath = os.path.join("results/single_agent_games", filename)
                
                with open(filepath, 'w') as f:
                    f.write(match['pgn'])
        
        print(f"Winning games saved to: results/single_agent_games/")


def main():
    """
    Main evaluation function for single-agent comparison
    """
    # Paths to 10M timestep models (agents 1, 2, 3)
    agents_10m = [
        "models/single_agent_1.zip",
        "models/single_agent_2.zip",
        "models/single_agent_3.zip"
    ]
    
    # Paths to 100K timestep models (agents 10, 11, 12)
    agents_100k = [
        "models/single_agent_10.zip",
        "models/single_agent_11.zip",
        "models/single_agent_12.zip"
    ]
    
    # Check if files exist
    all_paths = agents_10m + agents_100k
    missing_files = []
    for path in all_paths:
        if not os.path.exists(path):
            missing_files.append(path)
            print(f"Warning: Model not found: {path}")
    
    if missing_files:
        print(f"\n{len(missing_files)} model file(s) missing!")
        print("Please ensure all models are trained and saved before evaluation.")
        return None
    
    # Create evaluator
    evaluator = SingleAgentEvaluator()
    
    # Run evaluation (100 matches per pairing with stochastic play, both colors)
    # 3 agents (10M) × 3 agents (100K) × 100 matches × 2 colors = 1800 total matches
    results = evaluator.evaluate_agents(
        agents_10m,
        agents_100k,
        n_matches=100
    )
    
    # Save results
    evaluator.save_results()
    
    return results


if __name__ == "__main__":
    main()
