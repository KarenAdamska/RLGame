"""Unit tests for the SpaceShooterEnv RL wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from space_shooter.env import SpaceShooterEnv
from space_shooter.game import Action


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_env(seed: int = 0) -> SpaceShooterEnv:
    return SpaceShooterEnv(seed=seed, render_mode=None)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestEnvInit:
    def test_action_space_size(self):
        env = make_env()
        assert env.action_space.n == len(Action)

    def test_observation_space_shape(self):
        env = make_env()
        assert len(env.observation_space.shape) == 1
        assert env.observation_space.shape[0] == env.game.observation_size

    def test_observation_space_dtype(self):
        env = make_env()
        assert env.observation_space.dtype == np.float32


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestEnvReset:
    def test_reset_returns_tuple(self):
        env = make_env()
        result = env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_reset_obs_shape(self):
        env = make_env()
        obs, info = env.reset()
        assert obs.shape == (env.game.observation_size,)

    def test_reset_obs_dtype(self):
        env = make_env()
        obs, info = env.reset()
        assert obs.dtype == np.float32

    def test_reset_info_keys(self):
        env = make_env()
        _, info = env.reset()
        for key in ("score", "lives", "frame", "level"):
            assert key in info

    def test_reset_with_new_seed(self):
        env = make_env(seed=1)
        obs1, _ = env.reset(seed=1)
        obs2, _ = env.reset(seed=1)
        np.testing.assert_array_equal(obs1, obs2)


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

class TestEnvStep:
    def test_step_returns_five_tuple(self):
        env = make_env()
        env.reset()
        result = env.step(Action.NOOP)
        assert len(result) == 5

    def test_step_obs_shape(self):
        env = make_env()
        env.reset()
        obs, *_ = env.step(Action.NOOP)
        assert obs.shape == (env.game.observation_size,)

    def test_step_obs_dtype(self):
        env = make_env()
        env.reset()
        obs, *_ = env.step(Action.NOOP)
        assert obs.dtype == np.float32

    def test_step_reward_is_float(self):
        env = make_env()
        env.reset()
        _, reward, *_ = env.step(Action.NOOP)
        assert isinstance(reward, float)

    def test_step_terminated_is_bool(self):
        env = make_env()
        env.reset()
        _, _, terminated, truncated, _ = env.step(Action.NOOP)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_step_info_keys(self):
        env = make_env()
        env.reset()
        *_, info = env.step(Action.NOOP)
        for key in ("score", "lives", "frame"):
            assert key in info

    def test_step_increments_score_on_kill(self):
        env = make_env()
        env.reset()
        from space_shooter.game import Bullet, Enemy
        # Place an enemy and bullet in collision
        env.game.enemies.append(Enemy(x=100.0, y=100.0))
        env.game.bullets.append(Bullet(x=100.0, y=100.0))
        _, reward, *_ = env.step(Action.NOOP)
        assert env.game.score == 1
        assert reward >= 10.0


# ---------------------------------------------------------------------------
# max_steps truncation
# ---------------------------------------------------------------------------

class TestMaxSteps:
    def test_truncated_when_max_steps_reached(self):
        env = SpaceShooterEnv(seed=0, render_mode=None, max_steps=5)
        env.reset()
        terminated = truncated = False
        for _ in range(5):
            _, _, terminated, truncated, _ = env.step(Action.NOOP)
        assert truncated or terminated

    def test_no_truncation_without_max_steps(self):
        env = SpaceShooterEnv(seed=0, render_mode=None, max_steps=0)
        env.reset()
        truncated_any = False
        for _ in range(50):
            _, _, terminated, truncated, _ = env.step(Action.NOOP)
            if truncated:
                truncated_any = True
                break
            if terminated:
                break
        assert not truncated_any


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestEnvDeterminism:
    def test_same_seed_same_trajectory(self):
        actions = [Action.SHOOT, Action.LEFT, Action.RIGHT, Action.NOOP] * 10
        env1 = SpaceShooterEnv(seed=7)
        env2 = SpaceShooterEnv(seed=7)
        env1.reset()
        env2.reset()
        for a in actions:
            obs1, r1, t1, tr1, _ = env1.step(a)
            obs2, r2, t2, tr2, _ = env2.step(a)
            np.testing.assert_array_equal(obs1, obs2)
            assert r1 == r2
            assert t1 == t2


# ---------------------------------------------------------------------------
# Close
# ---------------------------------------------------------------------------

class TestEnvClose:
    def test_close_does_not_raise(self):
        env = make_env()
        env.reset()
        env.close()  # should not raise


# ---------------------------------------------------------------------------
# Render (rgb_array mode, headless)
# ---------------------------------------------------------------------------

class TestEnvRender:
    def test_rgb_array_shape(self):
        env = SpaceShooterEnv(seed=0, render_mode="rgb_array")
        env.reset()
        frame = env.render()
        assert frame is not None
        from space_shooter.game import SCREEN_HEIGHT, SCREEN_WIDTH
        assert frame.shape == (SCREEN_HEIGHT, SCREEN_WIDTH, 3)
        assert frame.dtype == np.uint8
        env.close()

    def test_none_render_mode_returns_none(self):
        env = SpaceShooterEnv(seed=0, render_mode=None)
        env.reset()
        result = env.render()
        assert result is None
        env.close()
