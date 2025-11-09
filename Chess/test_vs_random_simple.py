"""
Test trained self-play agent against random opponent
"""
import numpy as np
from pettingzoo.classic import chess_v6
from sb3_contrib import MaskablePPO
import sys
from torch.distributions import Distribution 
Distribution.set_default_validate_args(False)

def test_agent_vs_random(model_path="./models/selfplay_agent.zip",model_path2="./models/selfplay_agent-100k.zip", n_games=100, render=False):
    """Test trained agent against random opponent"""
    

    # Load trained model
    try:
        model = MaskablePPO.load(model_path)
        model2 = MaskablePPO.load(model_path2)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Statistics
    agent_wins = 0
    agent_losses = 0
    draws = 0
    total_moves = []
    
    for game in range(1, n_games + 1):
        # Create environment
        if render and game == 1:
            env = chess_v6.env(render_mode="human")
        else:
            env = chess_v6.env(render_mode=None)
        
        env.reset()
        move_count = 0
        
        # Play game
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                env.step(None)
                continue
            
            # Get action mask
            if obs and 'action_mask' in obs:
                action_mask = obs['action_mask']
                legal_moves = np.where(action_mask)[0]
                
                if len(legal_moves) == 0:
                    env.step(None)
                    continue
                
                # Agent plays as player_0 (white), random plays as player_1 (black)
                if agent == 'player_0':
                    # Trained agent
                    obs_flat = obs['observation'].flatten().astype(np.float32)
                    action, _ = model.predict(obs_flat, action_masks=action_mask.astype(bool), deterministic=False)
                else:
                    obs_flat = obs['observation'].flatten().astype(np.float32)
                    action, _ = model2.predict(obs_flat, action_masks=action_mask.astype(bool), deterministic=False)
                
                env.step(action)
                move_count += 1
            else:
                env.step(None)
        
        # Game ended - check result
        rewards = env.rewards
        agent_reward = rewards.get('player_0', 0)
        
        if agent_reward == 1:
            agent_wins += 1
            outcome = "WIN"
        elif agent_reward == -1:
            agent_losses += 1
            outcome = "LOSS"
        else:
            draws += 1
            outcome = "DRAW"
        
        total_moves.append(move_count)
        
        if game % 10 == 0 or game == 1:
            win_rate = agent_wins / game * 100
            print(f"Game {game:3d}: {outcome:5s} | Moves: {move_count:3d} | "
                  f"W:{agent_wins:3d} D:{draws:3d} L:{agent_losses:3d} | "
                  f"Win Rate: {win_rate:5.1f}%")
        
        env.close()
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Total Games: {n_games}")
    print(f"\nAgent (White) Results:")
    print(f"  Wins:   {agent_wins:4d} ({agent_wins/n_games*100:5.1f}%)")
    print(f"  Draws:  {draws:4d} ({draws/n_games*100:5.1f}%)")
    print(f"  Losses: {agent_losses:4d} ({agent_losses/n_games*100:5.1f}%)")
    print(f"\nAverage moves per game: {np.mean(total_moves):.1f}")
    print(f"Min moves: {np.min(total_moves)}")
    print(f"Max moves: {np.max(total_moves)}")
    
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    
    win_rate = agent_wins / n_games * 100
    
    if win_rate > 70:
        print("✓ EXCELLENT! Agent dominates random opponent")
    elif win_rate > 50:
        print("✓ GOOD! Agent beats random opponent consistently")
    elif win_rate > 30:
        print("⚠ MODERATE: Agent has some advantage over random")
    else:
        print("✗ POOR: Agent struggles against random opponent")
        print("  → May need more training or better reward shaping")
    
    print("="*60)
    
    return {
        'wins': agent_wins,
        'draws': draws,
        'losses': agent_losses,
        'win_rate': win_rate,
        'avg_moves': np.mean(total_moves)
    }


if __name__ == "__main__":
    # Parse command line arguments
    model_path = sys.argv[1] if len(sys.argv) > 1 else "./models/selfplay_agent.zip"
    model_path2 = sys.argv[2] if len(sys.argv) > 2 else "./models/selfplay_agent-100k.zip"
    n_games = int(sys.argv[2]) if len(sys.argv) > 3 else 1000
    render = sys.argv[3] == "render" if len(sys.argv) >4 else False
    
    results = test_agent_vs_random(model_path=model_path,model_path2=model_path2, n_games=n_games, render=render)
