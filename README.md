# RLGame — Space Shooter

A space shooter game built for reinforcement learning.

## Features

- **Decoupled game logic** – `space_shooter/game.py` is pure Python with no rendering
  dependency, so RL agents can train at full speed.
- **Gymnasium-compatible environment** – `space_shooter/env.py` exposes the standard
  `reset() / step() / render() / close()` API.
- **Deterministic with seeding** – pass `seed=N` to reproduce any episode exactly.
- **Rich observation vector** – normalised float32 array covering player state, nearby
  enemies, and active bullets.
- **Pygame renderer** – optional visual rendering for human play or debugging.

## Quickstart

### Install dependencies

```bash
pip install -r requirements.txt
```

### Play manually

```bash
python main.py
```

| Key | Action |
|-----|--------|
| ← / A | Move left |
| → / D | Move right |
| Space | Shoot |
| R | Restart (after game over) |
| Q / Escape | Quit |

### Use as an RL environment

```python
from space_shooter import SpaceShooterEnv

env = SpaceShooterEnv(seed=42)
obs, info = env.reset()

done = False
while not done:
    action = env.action_space.sample()   # replace with your agent
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
print("Final score:", info["score"])
```

### Render frames as numpy arrays (e.g., for visual RL)

```python
env = SpaceShooterEnv(seed=0, render_mode="rgb_array")
env.reset()
frame = env.render()   # shape: (800, 600, 3), dtype: uint8
```

## Environment specification

| Item | Value |
|------|-------|
| Observation space | `Box(-1, 1, shape=(43,), dtype=float32)` |
| Action space | `Discrete(4)` — 0 NOOP · 1 LEFT · 2 RIGHT · 3 SHOOT |
| Reward per step | +10 destroy enemy · −1 enemy escapes · −10 player hit · +0.1 alive |
| Episode end | Player lives reach 0 |

## Project layout

```
space_shooter/
  __init__.py   – package exports
  game.py       – core game logic (no rendering)
  renderer.py   – optional pygame rendering
  env.py        – Gymnasium-compatible RL environment
main.py         – interactive play entry point
tests/
  test_game.py  – unit tests for game logic
  test_env.py   – unit tests for RL environment
requirements.txt
```

## Running tests

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python -m pytest tests/ -v
```
