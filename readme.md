#  RL Training

This project contains reinforcement learning implementations for humanoid locomotion using various algorithms.

## Setup

### Environment Setup
Create a new conda environment with Python 3.9:
```bash
conda create -n rl_env python=3.9 -y
conda activate rl_env
```

### Install Dependencies
install from requirements.txt:
```bash
pip install -r requirements.txt
```

### Verify Installation
Test that MuJoCo and Humanoid environment work:
```bash
python -c "import gymnasium as gym; env = gym.make('Humanoid-v5'); print('Setup successful!')"
```

## Usage

### Single Agent Training
Run 5 training sessions and store results:
```bash
cd Humanoid
python humanoid_single_agent.py
```
To test, uncomment test_model() and comment main. (render_mode="human") for UI rendering

This will:
- Train 5 separate SAC agents for 100k timesteps each
- Store models in `models/standard_rl/run_X/`
- Store training logs in `logs/standard_rl/run_X/`
- Save results summary in `results/all_runs_summary_TIMESTAMP.csv`
### HMARL Agent Training
```bash
python train_hmarl.py
python test_hmarl.py
```


## Chess
Check the readme inside Chess folder.