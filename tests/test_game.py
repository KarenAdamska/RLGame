"""Unit tests for the core SpaceShooterGame logic."""

from __future__ import annotations

import pytest

from space_shooter.game import (
    BULLET_COOLDOWN,
    PLAYER_START_LIVES,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    Action,
    SpaceShooterGame,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_game(seed: int = 0) -> SpaceShooterGame:
    return SpaceShooterGame(seed=seed)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_player_starts_centered(self):
        game = make_game()
        assert game.player.x == pytest.approx((SCREEN_WIDTH - game.player.width) / 2)

    def test_player_starts_at_bottom(self):
        game = make_game()
        assert game.player.y == SCREEN_HEIGHT - game.player.height - 10

    def test_initial_lives(self):
        game = make_game()
        assert game.player.lives == PLAYER_START_LIVES

    def test_no_enemies_or_bullets_at_start(self):
        game = make_game()
        assert game.enemies == []
        assert game.bullets == []

    def test_score_zero_at_start(self):
        game = make_game()
        assert game.score == 0

    def test_done_false_at_start(self):
        game = make_game()
        assert not game.done


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_state(self):
        game = make_game()
        # play a few frames to dirty state
        for _ in range(20):
            game.step(Action.SHOOT)
        game.reset()
        assert game.score == 0
        assert game.frame == 0
        assert game.player.lives == PLAYER_START_LIVES
        assert not game.done

    def test_reset_returns_observation(self):
        game = make_game()
        obs = game.reset()
        assert isinstance(obs, list)
        assert len(obs) == game.observation_size


# ---------------------------------------------------------------------------
# Player movement
# ---------------------------------------------------------------------------

class TestPlayerMovement:
    def test_move_left_decreases_x(self):
        game = make_game()
        initial_x = game.player.x
        game.step(Action.LEFT)
        assert game.player.x < initial_x

    def test_move_right_increases_x(self):
        game = make_game()
        initial_x = game.player.x
        game.step(Action.RIGHT)
        assert game.player.x > initial_x

    def test_player_stays_within_left_bound(self):
        game = make_game()
        game.player.x = 0
        for _ in range(5):
            game.step(Action.LEFT)
        assert game.player.x >= 0

    def test_player_stays_within_right_bound(self):
        game = make_game()
        game.player.x = SCREEN_WIDTH - game.player.width
        for _ in range(5):
            game.step(Action.RIGHT)
        assert game.player.x <= SCREEN_WIDTH - game.player.width

    def test_noop_does_not_move_player(self):
        game = make_game()
        initial_x = game.player.x
        game.step(Action.NOOP)
        assert game.player.x == initial_x


# ---------------------------------------------------------------------------
# Shooting
# ---------------------------------------------------------------------------

class TestShooting:
    def test_shoot_creates_bullet(self):
        game = make_game()
        game.step(Action.SHOOT)
        assert len(game.bullets) == 1

    def test_shoot_respects_cooldown(self):
        game = make_game()
        game.step(Action.SHOOT)
        assert len(game.bullets) == 1
        # Shoot again immediately – should be blocked by cooldown
        game.step(Action.SHOOT)
        assert len(game.bullets) == 1

    def test_shoot_cooldown_expires(self):
        game = make_game()
        game.step(Action.SHOOT)
        # Advance enough frames to exhaust cooldown
        for _ in range(BULLET_COOLDOWN + 1):
            game.step(Action.NOOP)
        game.step(Action.SHOOT)
        assert len(game.bullets) == 2

    def test_bullet_moves_upward(self):
        game = make_game()
        game.step(Action.SHOOT)
        initial_y = game.bullets[0].y
        game.step(Action.NOOP)
        assert game.bullets[0].y < initial_y

    def test_bullet_removed_when_off_screen(self):
        game = make_game()
        game.step(Action.SHOOT)
        # Force bullet above screen
        game.bullets[0].y = -game.bullets[0].height - 1
        game.step(Action.NOOP)
        assert len(game.bullets) == 0


# ---------------------------------------------------------------------------
# Enemies
# ---------------------------------------------------------------------------

class TestEnemies:
    def test_enemies_spawn_over_time(self):
        game = make_game()
        # Run many frames to guarantee at least one spawn
        for _ in range(100):
            obs, reward, done = game.step(Action.NOOP)
            if game.enemies or done:
                break
        assert len(game.enemies) > 0 or game.done  # game might end

    def test_enemy_moves_downward(self):
        game = make_game()
        # Force an enemy at the top
        from space_shooter.game import Enemy
        game.enemies.append(Enemy(x=100.0, y=0.0))
        initial_y = game.enemies[-1].y
        game.step(Action.NOOP)
        assert game.enemies[-1].y > initial_y

    def test_enemy_removed_when_past_bottom(self):
        game = make_game()
        from space_shooter.game import Enemy
        game.enemies.append(Enemy(x=100.0, y=SCREEN_HEIGHT + 10))
        game.step(Action.NOOP)
        assert len(game.enemies) == 0

    def test_enemy_escape_penalises_reward(self):
        game = make_game()
        from space_shooter.game import Enemy
        game.enemies.append(Enemy(x=100.0, y=SCREEN_HEIGHT + 10))
        _, reward, _ = game.step(Action.NOOP)
        assert reward < 0  # net penalty


# ---------------------------------------------------------------------------
# Collision
# ---------------------------------------------------------------------------

class TestCollision:
    def test_bullet_destroys_enemy(self):
        game = make_game()
        from space_shooter.game import Bullet, Enemy
        # Place enemy and bullet at the same location
        game.enemies.append(Enemy(x=100.0, y=100.0))
        game.bullets.append(Bullet(x=100.0, y=100.0))
        _, reward, _ = game.step(Action.NOOP)
        assert len(game.enemies) == 0
        assert len(game.bullets) == 0
        assert reward >= 10.0  # destroy reward included

    def test_bullet_destroy_increases_score(self):
        game = make_game()
        from space_shooter.game import Bullet, Enemy
        game.enemies.append(Enemy(x=100.0, y=100.0))
        game.bullets.append(Bullet(x=100.0, y=100.0))
        game.step(Action.NOOP)
        assert game.score == 1

    def test_player_hit_reduces_lives(self):
        game = make_game()
        from space_shooter.game import Enemy
        # Place enemy directly on player
        game.enemies.append(
            Enemy(x=game.player.x, y=game.player.y)
        )
        game.step(Action.NOOP)
        assert game.player.lives == PLAYER_START_LIVES - 1

    def test_player_hit_gives_negative_reward(self):
        game = make_game()
        from space_shooter.game import Enemy
        game.enemies.append(
            Enemy(x=game.player.x, y=game.player.y)
        )
        _, reward, _ = game.step(Action.NOOP)
        assert reward < 0


# ---------------------------------------------------------------------------
# Game over
# ---------------------------------------------------------------------------

class TestGameOver:
    def test_game_over_when_lives_depleted(self):
        game = make_game()
        from space_shooter.game import Enemy
        game.player.lives = 1
        game.enemies.append(Enemy(x=game.player.x, y=game.player.y))
        _, _, done = game.step(Action.NOOP)
        assert done
        assert game.done

    def test_step_after_game_over_returns_done(self):
        game = make_game()
        game.done = True
        _, reward, done = game.step(Action.NOOP)
        assert done
        assert reward == 0.0


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class TestObservation:
    def test_observation_length(self):
        game = make_game()
        obs = game.get_observation()
        assert len(obs) == game.observation_size

    def test_observation_all_floats(self):
        game = make_game()
        obs = game.get_observation()
        assert all(isinstance(v, float) for v in obs)

    def test_observation_player_x_in_range(self):
        game = make_game()
        obs = game.get_observation()
        # Player x normalised to [0, 1]
        assert 0.0 <= obs[0] <= 1.0

    def test_observation_lives_in_range(self):
        game = make_game()
        obs = game.get_observation()
        assert 0.0 <= obs[2] <= 1.0

    def test_observation_changes_after_step(self):
        game = make_game()
        obs_before = game.get_observation()
        game.step(Action.LEFT)
        obs_after = game.get_observation()
        assert obs_before != obs_after


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_trajectory(self):
        actions = [Action.SHOOT, Action.LEFT, Action.RIGHT, Action.NOOP] * 30
        game1 = SpaceShooterGame(seed=42)
        game2 = SpaceShooterGame(seed=42)
        for a in actions:
            obs1, r1, d1 = game1.step(a)
            obs2, r2, d2 = game2.step(a)
            assert obs1 == obs2
            assert r1 == r2
            assert d1 == d2

    def test_different_seeds_can_differ(self):
        game1 = SpaceShooterGame(seed=0)
        game2 = SpaceShooterGame(seed=999)
        for _ in range(200):
            game1.step(Action.NOOP)
            game2.step(Action.NOOP)
        # Enemy positions are driven by RNG; states should eventually differ.
        assert (
            game1.get_observation() != game2.get_observation()
            or game1.score != game2.score
        )
