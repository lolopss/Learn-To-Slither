# Learn-To-Slither

Simple reinforcement-learning Snake project using tabular Q-learning and a small, explainable state encoding.  
The project trains an agent to play a Snake variant (green apples = +, red apples = -) and provides a Pygame visualizer and small UI.

## What this project does
- Implements the game logic and rendering in [game.py](game.py) (`SnakeGame`).
- Encodes a compact vision-based state in [`IA_Snake.encode_state`](IA_Snake.py).
- Trains a tabular Q-learning agent (`Q_table`) and persists learning state to `learning_state/` via [`IA_Snake.train_agent`](IA_Snake.py).
- Visualizes per-step info and Q-values with [`IA_visualisation.draw_right_panel`](IA_visualisation.py) and the Pygame UI.
- Provides a small lobby UI ([lobby.run_lobby](lobby.py)) to Play / Train / Quit.
- Small utilities to plot episode progress: [graph.py](graph.py).

## Quick start / requirements
- Python 3.8+
- Required packages: numpy, pygame, matplotlib
Install requirements:
```sh
pip install numpy pygame matplotlib
```

## Run the program
Show CLI help (recommended):
```sh
python main.py -h
```
Start the lobby (default behavior):
```sh
python main.py
```
From the lobby:
- Press P to Play one game (calls [`IA_Snake.play_single_game`](IA_Snake.py)).
- Press T to start training (calls [`IA_Snake.train_agent`](IA_Snake.py)).
- Press Q or ESC to quit.

You can also pass options to the lobby via CLI arguments:
- `-qtable PATH`  — path to Q-table to load/save (default uses `-save` fallback).
- `-episode N`   — number of episodes to pass to training.
- `-visual on|off` — enable or disable visualization for periodic visual demos.
- `-speed N`     — visualizer FPS.

Example:
```sh
python main.py -qtable learning_state/q_table_10000.npy -episode 10001 -visual off -speed 10
```

## Training / Saved state
- Q-tables are saved as .npy files in `learning_state/` (e.g. `learning_state/q_table.npy`, `q_table_10000.npy`). See [`IA_Snake.train_agent`](IA_Snake.py).
- Per-episode final lengths are saved to `learning_state/episode_lengths.npy` when training finishes.

## Inspect training progress
Use the plotting helper:
```sh
python graph.py --path learning_state/episode_lengths.npy --smooth 50 --out out_dir/
```
- [`graph.py`](graph.py) will load the array, optionally smooth it, and save/show a PNG.

Quick check script:
- [oui.py](oui.py) loads `learning_state/episode_lengths.npy` and prints summary lines.

## Important modules & entry points
- [main.py](main.py) — program entry, builds CLI and launches [lobby.run_lobby](lobby.py).
- [lobby.run_lobby](lobby.py) — Pygame menu to Play / Train.
- [`IA_Snake.encode_state`](IA_Snake.py) — state encoding used by the agent.
- [`IA_Snake.train_agent`](IA_Snake.py) — training loop, saves Q-table and episode lengths.
- [`IA_Snake.play_single_game`](IA_Snake.py) — visual play / replay UI.
- [game.SnakeGame](game.py) — game rules and mechanics.
- [IA_visualisation.draw_right_panel](IA_visualisation.py) — UI for vision and Q-values.
- [graph.main](graph.py) — plotting utility.

## Notes & tips
- The state space is compact (4 directions × 3 bits each) as described in [`IA_Snake.encode_state`](IA_Snake.py).
- Use `-visual off` or run training from the lobby to speed up long training runs.
- If a Q-table file is missing, the code falls back to a zeroed table (see `_load_q` in [lobby.py](lobby.py)).

For detailed behavior and configuration, read the source files linked above.