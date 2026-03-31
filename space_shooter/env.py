"""Gymnasium-compatible environment wrapper for the Space Shooter game.

Usage example::

    from space_shooter import SpaceShooterEnv

    env = SpaceShooterEnv()
    obs, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()   # random agent
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    env.close()
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .game import (
    Action,
    OBS_MAX_BULLETS,
    OBS_MAX_ENEMIES,
    PLAYER_START_LIVES,
    SpaceShooterGame,
)

# ---------------------------------------------------------------------------
# Optional gymnasium import (falls back to a minimal shim if not installed)
# ---------------------------------------------------------------------------

try:
    import gymnasium as gym  # type: ignore
    from gymnasium import spaces  # type: ignore
    _GYM_BASE = gym.Env
except ImportError:  # pragma: no cover
    try:
        import gym  # type: ignore  # legacy gym
        from gym import spaces  # type: ignore
        _GYM_BASE = gym.Env
    except ImportError:
        # Provide a minimal stand-alone base so the env works without gym
        spaces = None  # type: ignore
        _GYM_BASE = object  # type: ignore


# ---------------------------------------------------------------------------
# Spaces shim (used when neither gymnasium nor gym is installed)
# ---------------------------------------------------------------------------

class _DiscreteSpace:
    """Minimal discrete action space shim."""

    def __init__(self, n: int) -> None:
        self.n = n

    def sample(self) -> int:
        import random
        return random.randrange(self.n)

    def contains(self, x: int) -> bool:
        return 0 <= x < self.n


class _BoxSpace:
    """Minimal continuous observation space shim."""

    def __init__(self, low: np.ndarray, high: np.ndarray) -> None:
        self.low = low
        self.high = high
        self.shape = low.shape
        self.dtype = low.dtype


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SpaceShooterEnv(_GYM_BASE):
    """Space Shooter Gymnasium environment.

    Observation space
    -----------------
    A 1-D float32 array of length ``3 + OBS_MAX_ENEMIES * 3 + OBS_MAX_BULLETS * 2``
    with values in [-1, 1]:

    * ``[0]``  – player x position, normalised to [0, 1]
    * ``[1]``  – shoot cooldown, normalised to [0, 1]
    * ``[2]``  – remaining lives, normalised to [0, 1]
    * ``[3 + i*3 ... 3 + i*3 + 2]`` – enemy *i*: (x, y, active), -1 if absent
    * ``[3 + N*3 + j*2 ... ]``      – bullet *j*: (x, y), -1 if absent

    Action space
    ------------
    Discrete(4):  0 = NOOP, 1 = LEFT, 2 = RIGHT, 3 = SHOOT

    Reward
    ------
    * ``+10`` per enemy destroyed
    * ``-1``  per enemy that escapes the bottom
    * ``-10`` when the player is hit
    * ``+0.1`` per step alive (survival bonus)

    Episode termination
    -------------------
    The episode ends when the player has no remaining lives.

    Parameters
    ----------
    seed:
        Optional integer seed for reproducibility.
    render_mode:
        ``"human"`` to open a pygame window, ``"rgb_array"`` to return pixel
        arrays from :meth:`render`, or ``None`` to disable rendering.
    max_steps:
        Maximum number of steps per episode (0 = unlimited).
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        max_steps: int = 0,
    ) -> None:
        super().__init__()

        self._seed = seed
        self._render_mode = render_mode
        self._max_steps = max_steps
        self._game = SpaceShooterGame(seed=seed)
        self._renderer: Any = None

        obs_size = self._game.observation_size
        low = np.full(obs_size, -1.0, dtype=np.float32)
        high = np.full(obs_size, 1.0, dtype=np.float32)

        if spaces is not None:
            self.action_space = spaces.Discrete(self._game.n_actions)
            self.observation_space = spaces.Box(
                low=low, high=high, shape=(obs_size,), dtype=np.float32
            )
        else:
            self.action_space = _DiscreteSpace(self._game.n_actions)
            self.observation_space = _BoxSpace(low, high)

        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and return (observation, info)."""
        if seed is not None:
            self._seed = seed
        self._game = SpaceShooterGame(seed=self._seed)
        self._step_count = 0
        obs = np.array(self._game.reset(), dtype=np.float32)
        info = self._build_info()
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Apply *action* and return ``(obs, reward, terminated, truncated, info)``."""
        raw_obs, reward, terminated = self._game.step(action)
        self._step_count += 1
        truncated = (
            self._max_steps > 0 and self._step_count >= self._max_steps
        )
        obs = np.array(raw_obs, dtype=np.float32)
        info = self._build_info()
        return obs, float(reward), terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Render the current game state.

        Returns an ``(H, W, 3)`` uint8 array when ``render_mode="rgb_array"``,
        or ``None`` for other modes.
        """
        if self._render_mode is None:
            return None
        renderer = self._get_renderer()
        return renderer.render(self._game, self._render_mode)

    def close(self) -> None:
        """Clean up resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def game(self) -> SpaceShooterGame:
        """Direct access to the underlying :class:`SpaceShooterGame`."""
        return self._game

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_info(self) -> Dict[str, Any]:
        return {
            "score": self._game.score,
            "lives": self._game.player.lives,
            "frame": self._game.frame,
            "level": self._game._level,
            "n_enemies": len(self._game.enemies),
            "n_bullets": len(self._game.bullets),
        }

    def _get_renderer(self) -> Any:
        if self._renderer is None:
            from .renderer import PygameRenderer  # lazy import
            self._renderer = PygameRenderer()
        return self._renderer
