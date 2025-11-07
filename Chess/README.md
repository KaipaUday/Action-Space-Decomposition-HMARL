# Chess Reinforcement Learning Training


## Files and folders

- `single_agent_trainer.py` - Single agent training

- `train_chess_hmarl.py` -  MARL training with decomposition
- `chess_hmarl/` - MARL module with environment and training code


Results are saved in:
- `models/chess_<method>_<timestamp>/` - Trained models 
    single_agent_10,11,12 are trained for 100k others are trained for 10million steps.
- `logs/chess_<method>_<timestamp>/` - TensorBoard logs
- `results/` - Comparison plots and statistics
## Usage

### Training

1. **Single Agent Training:**
```bash
cd Chess
python train_chess_single.py
```
Set the training timesteps ,comment testing and  set Chess_v6(render_mode="none") since UI redndering takes time.


2. **MARL Training:**
```bash
python train_chess_hmarl.py

python train_chess_hmarl.py test 1 #testing

```

### Comparing Results

After training all approaches, compare results:
```bash
python evaluate_agents.py #for single and MARL 
python eval_single_agents.py # for single 10m and 100k.
```



