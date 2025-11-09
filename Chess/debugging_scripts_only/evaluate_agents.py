"""
Evaluate and compare Single-Agent vs MARL Chess Agents
Plays matches between different agent types and reports results
"""
import numpy as np
from pettingzoo.classic import chess_v6
from stable_baselines3 import PPO
import json
from datetime import datetime
import os


class ChessEvaluator:
    def __init__(self):
        self.results = {
            'matches': [],
            'summary': {}
        }
    
    def play_match(self, white_model, black_model, white_name, black_name, match_id, 
                   white_is_marl=False, black_is_marl=False):
        """Play a single chess match between two agents"""
        env = chess_v6.env(render_mode=None)
        env.reset()
        
        move_count = 0
        max_moves = 250
        white_move_count = 0
        black_move_count = 0
        
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
            if current_player == 'player_0':  # White
                if white_is_marl:
                    # Select MARL agent based on move count
                    if white_move_count <= 15:
                        model = white_model['opening']
                    elif white_move_count <= 40:
                        model = white_model['middle']
                    else:
                        model = white_model['endgame']
                else:
                    model = white_model
                white_move_count += 1
            else:  # Black
                if black_is_marl:
                    # Select MARL agent based on move count
                    if black_move_count <= 15:
                        model = black_model['opening']
                    elif black_move_count <= 40:
                        model = black_model['middle']
                    else:
                        model = black_model['endgame']
                else:
                    model = black_model
                black_move_count += 1
            
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
    
    def evaluate_agents(self, single_agent_paths, marl_agent_sets, n_matches=10):
        """
        Evaluate single-agent vs MARL agents
        
        Args:
            single_agent_paths: List of paths to single-agent models (3 models)
            marl_agent_sets: List of MARL agent sets, each containing [opening, middle, endgame] paths
                            e.g., [['run1/opening.zip', 'run1/middle.zip', 'run1/endgame.zip'], ...]
            n_matches: Number of matches per pairing
        """
        print(f"Loading {len(single_agent_paths)} single-agent models...")
        single_agents = [PPO.load(path) for path in single_agent_paths]
        
        print(f"Loading {len(marl_agent_sets)} MARL agent sets (3 agents each)...")
        marl_agents = []
        for i, agent_set in enumerate(marl_agent_sets):
            print(f"  Loading MARL set {i+1}: opening, middle, endgame agents...")
            marl_agents.append({
                'opening': PPO.load(agent_set[0]),
                'middle': PPO.load(agent_set[1]),
                'endgame': PPO.load(agent_set[2])
            })
        
        total_matches = len(single_agents) * len(marl_agents) * n_matches * 2  # *2 for color swap
        match_count = 0
        
        print(f"\nStarting evaluation: {total_matches} total matches")
        print("=" * 60)
        
        # Play all combinations
        for i, single_agent in enumerate(single_agents):
            for j, marl_agent in enumerate(marl_agents):
                single_name = f"Single_{i+1}"
                marl_name = f"MARL_{j+1}"
                
                # Play n_matches with single-agent as white
                for match in range(n_matches):
                    match_count += 1
                    result = self.play_match(
                        single_agent, marl_agent,
                        single_name, marl_name,
                        match_count,
                        white_is_marl=False, black_is_marl=True
                    )
                    self.results['matches'].append(result)
                    
                    if match_count % 10 == 0:
                        print(f"Completed {match_count}/{total_matches} matches...")
                
                # Play n_matches with MARL as white (color swap)
                for match in range(n_matches):
                    match_count += 1
                    result = self.play_match(
                        marl_agent, single_agent,
                        marl_name, single_name,
                        match_count,
                        white_is_marl=True, black_is_marl=False
                    )
                    self.results['matches'].append(result)
                    
                    if match_count % 10 == 0:
                        print(f"Completed {match_count}/{total_matches} matches...")
        
        print("=" * 60)
        print("Evaluation complete!")
        
        # Calculate summary statistics
        self._calculate_summary()
        self._print_results()
        
        return self.results
    
    def _calculate_summary(self):
        """Calculate win/loss/draw statistics"""
        single_wins = 0
        marl_wins = 0
        draws = 0
        
        single_material_advantage = []
        marl_material_advantage = []
        
        for match in self.results['matches']:
            material_diff = match.get('final_material_diff', 0)
            
            if match['result'] == 'white_win':
                if 'Single' in match['white']:
                    single_wins += 1
                    single_material_advantage.append(material_diff)
                else:
                    marl_wins += 1
                    marl_material_advantage.append(-material_diff)
            elif match['result'] == 'black_win':
                if 'Single' in match['black']:
                    single_wins += 1
                    single_material_advantage.append(-material_diff)
                else:
                    marl_wins += 1
                    marl_material_advantage.append(material_diff)
            else:
                draws += 1
        
        total = len(self.results['matches'])
        
        self.results['summary'] = {
            'total_matches': total,
            'single_agent_wins': single_wins,
            'marl_wins': marl_wins,
            'draws': draws,
            'single_agent_win_rate': single_wins / total * 100,
            'marl_win_rate': marl_wins / total * 100,
            'draw_rate': draws / total * 100,
            'single_avg_material_in_wins': np.mean(single_material_advantage) if single_material_advantage else 0,
            'marl_avg_material_in_wins': np.mean(marl_material_advantage) if marl_material_advantage else 0
        }
    
    def _print_results(self):
        """Print formatted results"""
        summary = self.results['summary']
        
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Total Matches: {summary['total_matches']}")
        print(f"\nSingle-Agent Wins: {summary['single_agent_wins']} ({summary['single_agent_win_rate']:.1f}%)")
        print(f"  Avg Material Advantage in Wins: {summary['single_avg_material_in_wins']:.1f}")
        print(f"\nMARL Wins: {summary['marl_wins']} ({summary['marl_win_rate']:.1f}%)")
        print(f"  Avg Material Advantage in Wins: {summary['marl_avg_material_in_wins']:.1f}")
        print(f"\nDraws: {summary['draws']} ({summary['draw_rate']:.1f}%)")
        print("=" * 60)
    
    def save_results(self, filename=None):
        """Save results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.json"
        
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
        os.makedirs("results/winning_games", exist_ok=True)
        
        winning_games = [m for m in self.results['matches'] if m['result'] in ['white_win', 'black_win']]
        
        print(f"\nSaving {len(winning_games)} winning games...")
        
        for match in winning_games:
            if 'pgn' in match:
                filename = f"match_{match['match_id']}_{match['white']}_vs_{match['black']}_{match['result']}.pgn"
                filepath = os.path.join("results/winning_games", filename)
                
                with open(filepath, 'w') as f:
                    f.write(match['pgn'])
        
        print(f"Winning games saved to: results/winning_games/")


def main():
    """
    Main evaluation function
    
    Update these paths to your trained models:
    """
    # Paths to single-agent models (3 models)
    single_agent_paths = [
        "models/single_agent_1.zip",
        "models/single_agent_2.zip",
        "models/single_agent_3.zip"
    ]
    
    # Paths to MARL agent sets (3 sets, each with 3 agents)
    # Each set contains [opening, middle, endgame] agents
    marl_agent_sets = [
        [
            "models/hmarl_opening_1.zip",
            "models/hmarl_middle_1.zip",
            "models/hmarl_endgame_1.zip"
        ],
        [
            "models/hmarl_opening_2.zip",
            "models/hmarl_middle_2.zip",
            "models/hmarl_endgame_2.zip"
        ],
        [
            "models/hmarl_opening_3.zip",
            "models/hmarl_middle_3.zip",
            "models/hmarl_endgame_3.zip"
        ]
    ]
    
    # Check if files exist
    all_paths = single_agent_paths + [path for agent_set in marl_agent_sets for path in agent_set]
    for path in all_paths:
        if not os.path.exists(path):
            print(f"Warning: Model not found: {path}")
    
    # Create evaluator
    evaluator = ChessEvaluator()
    
    # Run evaluation (1000 matches per pairing with stochastic play, both colors)
    # 3 single-agent × 3 MARL sets × 1000 matches × 2 colors = 18000 total matches
    results = evaluator.evaluate_agents(
        single_agent_paths,
        marl_agent_sets,
        n_matches=1000
    )
    
    # Save results
    evaluator.save_results()
    
    return results


if __name__ == "__main__":
    main()
