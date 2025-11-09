"""
Test random vs random baseline and trained model vs random
Includes material difference tracking
"""
import numpy as np
from pettingzoo.classic import chess_v6
from sb3_contrib import MaskablePPO
import chess
import sys


def calculate_material(board):
    """Calculate material balance (positive = white ahead)"""
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    
    white_material = sum(len(board.pieces(piece, chess.WHITE)) * value 
                        for piece, value in piece_values.items())
    black_material = sum(len(board.pieces(piece, chess.BLACK)) * value 
                        for piece, value in piece_values.items())
    
    return white_material - black_material

from torch.distributions import Distribution 
Distribution.set_default_validate_args(False)

def test_random_vs_random(n_games=10000):
    """Test random agent vs random agent"""
    print("="*60)
    print("Random vs Random Baseline")
    print("="*60)
    print(f"Games: {n_games}")
    print("="*60)
    
    white_wins = 0
    black_wins = 0
    draws = 0
    total_moves = []
    final_material_diffs = []  # Track final material difference
    
    for game in range(1, n_games + 1):
        env = chess_v6.env(render_mode=None)
        env.reset()
        
        move_count = 0
        max_moves = 200
        
        # Play game
        for _ in range(max_moves * 2):  # 2 moves per turn
            if not env.agents:
                break
            
            obs, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                env.step(None)
                continue
            
            # Random move
            if obs and 'action_mask' in obs:
                legal = np.where(obs['action_mask'])[0]
                if len(legal) > 0:
                    action = np.random.choice(legal)
                    env.step(action)
                    move_count += 1
                else:
                    env.step(None)
            else:
                env.step(None)
        
        # Check result and material
        rewards = env.rewards
        white_reward = rewards.get('player_0', 0)
        
        # Calculate final material difference
        final_board = env.unwrapped.board
        material_diff = calculate_material(final_board)
        final_material_diffs.append(material_diff)
        
        if white_reward == 1:
            white_wins += 1
            outcome = "WHITE"
        elif white_reward == -1:
            black_wins += 1
            outcome = "BLACK"
        else:
            draws += 1
            outcome = "DRAW"
        
        total_moves.append(move_count)
        
        if game % 1000 == 0:
            print(f"Game {game:5d} | W:{white_wins:4d} B:{black_wins:4d} D:{draws:4d} | "
                  f"W%:{white_wins/game*100:5.1f} B%:{black_wins/game*100:5.1f} D%:{draws/game*100:5.1f}")
        
        env.close()
    
    # Print summary
    print("\n" + "="*60)
    print("RANDOM VS RANDOM RESULTS")
    print("="*60)
    print(f"Total Games: {n_games}")
    print(f"\nWhite (Random): {white_wins:5d} ({white_wins/n_games*100:5.1f}%)")
    print(f"Black (Random): {black_wins:5d} ({black_wins/n_games*100:5.1f}%)")
    print(f"Draws:          {draws:5d} ({draws/n_games*100:5.1f}%)")
    print(f"\nGame Statistics:")
    print(f"  Average moves: {np.mean(total_moves):.1f}")
    print(f"  Min moves: {np.min(total_moves)}")
    print(f"  Max moves: {np.max(total_moves)}")
    print(f"\nMaterial Difference (Final):")
    print(f"  Average: {np.mean(final_material_diffs):+.2f}")
    print(f"  Std Dev: {np.std(final_material_diffs):.2f}")
    print(f"  Min: {np.min(final_material_diffs):+.0f}")
    print(f"  Max: {np.max(final_material_diffs):+.0f}")
    print("="*60)
    
    return {
        'white_wins': white_wins,
        'black_wins': black_wins,
        'draws': draws,
        'avg_moves': np.mean(total_moves),
        'avg_material_diff': np.mean(final_material_diffs),
        'material_diffs': final_material_diffs
    }


def test_model_vs_random(model_path, n_games=10000):
    """Test trained model vs random agent"""
    print("\n" + "="*60)
    print("Trained Model vs Random")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Games: {n_games}")
    print("="*60)
    
    # Load model
    try:
        model = MaskablePPO.load(model_path)
        print("✓ Model loaded successfully\n")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None
    
    agent_wins = 0
    agent_losses = 0
    draws = 0
    total_moves = []
    final_material_diffs = []  # Track final material difference
    
    # Track material advantage when winning/losing
    win_material_diffs = []
    loss_material_diffs = []
    
    for game in range(1, n_games + 1):
        env = chess_v6.env(render_mode=None)
        env.reset()
        
        move_count = 0
        
        # Play game
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                env.step(None)
                continue
            
            if obs and 'action_mask' in obs:
                action_mask = obs['action_mask']
                legal = np.where(action_mask)[0]
                
                if len(legal) == 0:
                    env.step(None)
                    continue
                
                # Agent plays as white (player_0), random as black (player_1)
                if agent == 'player_0':
                    # Trained model
                    obs_flat = obs['observation'].flatten().astype(np.float32)
                    action, _ = model.predict(obs_flat, action_masks=action_mask.astype(bool), deterministic=False)
                else:
                    # Random opponent
                    action = np.random.choice(legal)
                
                env.step(action)
                move_count += 1
            else:
                env.step(None)
        
        # Check result and material
        rewards = env.rewards
        agent_reward = rewards.get('player_0', 0)
        
        # Calculate final material difference
        final_board = env.unwrapped.board
        material_diff = calculate_material(final_board)
        final_material_diffs.append(material_diff)
        
        if agent_reward == 1:
            agent_wins += 1
            outcome = "WIN"
            win_material_diffs.append(material_diff)
        elif agent_reward == -1:
            agent_losses += 1
            outcome = "LOSS"
            loss_material_diffs.append(material_diff)
        else:
            draws += 1
            outcome = "DRAW"
        
        total_moves.append(move_count)
        
        if game % 1000 == 0:
            win_rate = agent_wins / game * 100
            print(f"Game {game:5d} | W:{agent_wins:4d} D:{draws:4d} L:{agent_losses:4d} | "
                  f"WR:{win_rate:5.1f}% | Avg:{np.mean(total_moves[-1000:]):5.1f} moves")
        
        env.close()
    
    # Print summary
    win_rate = agent_wins / n_games * 100
    draw_rate = draws / n_games * 100
    loss_rate = agent_losses / n_games * 100
    
    print("\n" + "="*60)
    print("TRAINED MODEL VS RANDOM RESULTS")
    print("="*60)
    print(f"Total Games: {n_games}")
    print(f"\nAgent (White): {agent_wins:5d} ({win_rate:5.1f}%)")
    print(f"Draws:         {draws:5d} ({draw_rate:5.1f}%)")
    print(f"Random (Black):{agent_losses:5d} ({loss_rate:5.1f}%)")
    print(f"\nGame Statistics:")
    print(f"  Average moves: {np.mean(total_moves):.1f}")
    print(f"  Min moves: {np.min(total_moves)}")
    print(f"  Max moves: {np.max(total_moves)}")
    print(f"\nMaterial Difference (Final):")
    print(f"  Overall Average: {np.mean(final_material_diffs):+.2f}")
    print(f"  Overall Std Dev: {np.std(final_material_diffs):.2f}")
    
    if len(win_material_diffs) > 0:
        print(f"\n  When Winning:")
        print(f"    Average material advantage: {np.mean(win_material_diffs):+.2f}")
        print(f"    Min: {np.min(win_material_diffs):+.0f}, Max: {np.max(win_material_diffs):+.0f}")
    
    if len(loss_material_diffs) > 0:
        print(f"\n  When Losing:")
        print(f"    Average material difference: {np.mean(loss_material_diffs):+.2f}")
        print(f"    Min: {np.min(loss_material_diffs):+.0f}, Max: {np.max(loss_material_diffs):+.0f}")
    
    # Performance evaluation
    print("\n" + "="*60)
    print("PERFORMANCE EVALUATION")
    print("="*60)
    
    if win_rate > 80:
        rating = "EXCELLENT"
    elif win_rate > 65:
        rating = "VERY GOOD"
    elif win_rate > 50:
        rating = "GOOD"
    elif win_rate > 35:
        rating = "MODERATE"
    else:
        rating = "POOR"
    
    print(f"Rating: {rating}")
    print(f"Win Rate: {win_rate:.1f}%")
    
    # Compare with random baseline (should be ~50%)
    improvement = win_rate - 50.0
    print(f"Improvement over random: {improvement:+.1f}%")
    
    print("="*60)
    
    return {
        'wins': agent_wins,
        'draws': draws,
        'losses': agent_losses,
        'win_rate': win_rate,
        'avg_moves': np.mean(total_moves),
        'avg_material_diff': np.mean(final_material_diffs),
        'win_material_avg': np.mean(win_material_diffs) if len(win_material_diffs) > 0 else 0,
        'loss_material_avg': np.mean(loss_material_diffs) if len(loss_material_diffs) > 0 else 0,
        'material_diffs': final_material_diffs
    }


def compare_both(model_path, n_games=10000):
    """Run both tests and compare"""
    print("\n" + "="*60)
    print("COMPREHENSIVE COMPARISON")
    print("="*60)
    print(f"Running {n_games} games for each test...")
    print("="*60)
    
    # Test 1: Random vs Random
    random_results = test_random_vs_random(n_games)
    
    # Test 2: Model vs Random
    model_results = test_model_vs_random(model_path, n_games)
    
    if model_results is None:
        return
    
    # Compare
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    print("\nRandom vs Random (Baseline):")
    print(f"  White wins: {random_results['white_wins']/n_games*100:5.1f}%")
    print(f"  Black wins: {random_results['black_wins']/n_games*100:5.1f}%")
    print(f"  Draws:      {random_results['draws']/n_games*100:5.1f}%")
    
    print("\nTrained Model vs Random:")
    print(f"  Model wins: {model_results['win_rate']:5.1f}%")
    print(f"  Draws:      {model_results['draws']/n_games*100:5.1f}%")
    print(f"  Losses:     {model_results['losses']/n_games*100:5.1f}%")
    
    print("\nImprovement:")
    baseline_white = random_results['white_wins']/n_games*100
    improvement = model_results['win_rate'] - baseline_white
    print(f"  Win Rate: {improvement:+.1f}%")
    
    print("\nMaterial Difference:")
    print(f"  Random baseline: {random_results['avg_material_diff']:+.2f}")
    print(f"  Trained model:   {model_results['avg_material_diff']:+.2f}")
    print(f"  Improvement:     {model_results['avg_material_diff'] - random_results['avg_material_diff']:+.2f}")
    
    if model_results['win_material_avg'] > 0:
        print(f"\n  Model wins with avg material advantage: {model_results['win_material_avg']:+.2f}")
    if model_results['loss_material_avg'] != 0:
        print(f"  Model loses with avg material difference: {model_results['loss_material_avg']:+.2f}")
    
    print("\nOverall Assessment:")
    if improvement > 30:
        print("  ✓ EXCELLENT: Model significantly outperforms random")
    elif improvement > 15:
        print("  ✓ GOOD: Model clearly learned useful strategies")
    elif improvement > 5:
        print("  ⚠ MODERATE: Model shows some learning")
    else:
        print("  ✗ POOR: Model barely better than random")
    
    print("="*60)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "random":
            # Test random vs random only
            n_games = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
            test_random_vs_random(n_games)
        elif sys.argv[1] == "model":
            # Test model vs random only
            model_path = sys.argv[2] if len(sys.argv) > 2 else "./models/selfplay_agent1m.zip"
            n_games = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
            test_model_vs_random(model_path, n_games)
        elif sys.argv[1] == "compare":
            # Compare both
            model_path = sys.argv[2] if len(sys.argv) > 2 else "./models/selfplay_agent1m.zip"
            n_games = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
            compare_both(model_path, n_games)
    else:
        # Default: compare both with your model
        compare_both("./models/selfplay_agent1m.zip", n_games=10000)
