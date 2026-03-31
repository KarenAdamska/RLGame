"""Core game logic for the Space Shooter game.

This module is intentionally free of any rendering dependencies so that
it can be imported and used by RL agents without a display or pygame.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCREEN_WIDTH: int = 600
SCREEN_HEIGHT: int = 800

PLAYER_WIDTH: int = 50
PLAYER_HEIGHT: int = 30
PLAYER_SPEED: int = 5
PLAYER_START_LIVES: int = 3

BULLET_WIDTH: int = 6
BULLET_HEIGHT: int = 16
BULLET_SPEED: int = 10
BULLET_COOLDOWN: int = 15  # frames between shots

ENEMY_WIDTH: int = 40
ENEMY_HEIGHT: int = 30
ENEMY_BASE_SPEED: float = 2.0
ENEMY_SPAWN_INTERVAL: int = 45  # frames between enemy spawns
MAX_ENEMIES: int = 20

REWARD_DESTROY_ENEMY: float = 10.0
REWARD_ENEMY_ESCAPED: float = -1.0
REWARD_PLAYER_HIT: float = -10.0
REWARD_STEP_ALIVE: float = 0.1

# Maximum number of enemies / bullets tracked in the observation vector.
OBS_MAX_ENEMIES: int = 10
OBS_MAX_BULLETS: int = 5


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Player:
    x: float
    y: float
    width: int = PLAYER_WIDTH
    height: int = PLAYER_HEIGHT
    lives: int = PLAYER_START_LIVES
    shoot_cooldown: int = 0

    @property
    def rect(self) -> Tuple[float, float, int, int]:
        """Return (x, y, width, height) bounding rect."""
        return (self.x, self.y, self.width, self.height)

    def center_x(self) -> float:
        return self.x + self.width / 2

    def center_y(self) -> float:
        return self.y + self.height / 2


@dataclass
class Bullet:
    x: float
    y: float
    width: int = BULLET_WIDTH
    height: int = BULLET_HEIGHT
    speed: int = BULLET_SPEED
    active: bool = True

    @property
    def rect(self) -> Tuple[float, float, int, int]:
        return (self.x, self.y, self.width, self.height)


@dataclass
class Enemy:
    x: float
    y: float
    width: int = ENEMY_WIDTH
    height: int = ENEMY_HEIGHT
    speed: float = ENEMY_BASE_SPEED
    active: bool = True

    @property
    def rect(self) -> Tuple[float, float, int, int]:
        return (self.x, self.y, self.width, self.height)

    def center_x(self) -> float:
        return self.x + self.width / 2

    def center_y(self) -> float:
        return self.y + self.height / 2


# ---------------------------------------------------------------------------
# Action enum
# ---------------------------------------------------------------------------

class Action(IntEnum):
    NOOP = 0
    LEFT = 1
    RIGHT = 2
    SHOOT = 3


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _rects_overlap(
    ax: float, ay: float, aw: int, ah: int,
    bx: float, by: float, bw: int, bh: int,
) -> bool:
    """Return True if two axis-aligned rectangles overlap."""
    return (
        ax < bx + bw
        and ax + aw > bx
        and ay < by + bh
        and ay + ah > by
    )


# ---------------------------------------------------------------------------
# Main game class
# ---------------------------------------------------------------------------

class SpaceShooterGame:
    """Pure-logic space shooter game.

    The game runs deterministically given a random seed, which makes it
    suitable for reproducible RL experiments.

    The observation returned by :meth:`get_observation` is a flat list of
    normalised floats that can be directly fed into a neural network.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> List[float]:
        """Reset the game to its initial state and return the first observation."""
        self.player = Player(
            x=(SCREEN_WIDTH - PLAYER_WIDTH) / 2,
            y=SCREEN_HEIGHT - PLAYER_HEIGHT - 10,
        )
        self.bullets: List[Bullet] = []
        self.enemies: List[Enemy] = []
        self.score: int = 0
        self.frame: int = 0
        self.done: bool = False
        self._spawn_timer: int = 0
        self._level: int = 1
        return self.get_observation()

    def step(self, action: int) -> Tuple[List[float], float, bool]:
        """Advance the game by one frame.

        Args:
            action: An integer from :class:`Action`.

        Returns:
            A tuple of ``(observation, reward, done)``.
        """
        if self.done:
            return self.get_observation(), 0.0, True

        reward = 0.0
        self.frame += 1

        # --- 1. Apply player action ---
        action = Action(action)
        if action == Action.LEFT:
            self.player.x = max(0, self.player.x - PLAYER_SPEED)
        elif action == Action.RIGHT:
            self.player.x = min(
                SCREEN_WIDTH - self.player.width,
                self.player.x + PLAYER_SPEED,
            )
        elif action == Action.SHOOT:
            if self.player.shoot_cooldown == 0:
                self._fire_bullet()
                self.player.shoot_cooldown = BULLET_COOLDOWN

        # Cool down shoot timer
        if self.player.shoot_cooldown > 0:
            self.player.shoot_cooldown -= 1

        # --- 2. Spawn enemies ---
        self._spawn_timer += 1
        interval = max(20, ENEMY_SPAWN_INTERVAL - (self._level - 1) * 5)
        if self._spawn_timer >= interval and len(self.enemies) < MAX_ENEMIES:
            self._spawn_enemy()
            self._spawn_timer = 0

        # Increase level every 500 frames
        self._level = 1 + self.frame // 500

        # --- 3. Move bullets ---
        for bullet in self.bullets:
            if bullet.active:
                bullet.y -= bullet.speed

        # --- 4. Move enemies ---
        enemy_speed = ENEMY_BASE_SPEED + (self._level - 1) * 0.5
        for enemy in self.enemies:
            if enemy.active:
                enemy.y += enemy_speed

        # --- 5. Check bullet-enemy collisions ---
        for bullet in self.bullets:
            if not bullet.active:
                continue
            for enemy in self.enemies:
                if not enemy.active:
                    continue
                if _rects_overlap(
                    bullet.x, bullet.y, bullet.width, bullet.height,
                    enemy.x, enemy.y, enemy.width, enemy.height,
                ):
                    bullet.active = False
                    enemy.active = False
                    self.score += 1
                    reward += REWARD_DESTROY_ENEMY

        # --- 6. Check player-enemy collisions ---
        for enemy in self.enemies:
            if not enemy.active:
                continue
            if _rects_overlap(
                self.player.x, self.player.y,
                self.player.width, self.player.height,
                enemy.x, enemy.y, enemy.width, enemy.height,
            ):
                enemy.active = False
                self.player.lives -= 1
                reward += REWARD_PLAYER_HIT

        # --- 7. Remove out-of-bounds bullets ---
        for bullet in self.bullets:
            if bullet.active and bullet.y + bullet.height < 0:
                bullet.active = False

        # --- 8. Remove enemies that reached the bottom ---
        for enemy in self.enemies:
            if enemy.active and enemy.y > SCREEN_HEIGHT:
                enemy.active = False
                reward += REWARD_ENEMY_ESCAPED

        # --- 9. Prune inactive objects ---
        self.bullets = [b for b in self.bullets if b.active]
        self.enemies = [e for e in self.enemies if e.active]

        # --- 10. Check game over ---
        if self.player.lives <= 0:
            self.done = True
        else:
            reward += REWARD_STEP_ALIVE

        return self.get_observation(), reward, self.done

    def get_observation(self) -> List[float]:
        """Return a flat, normalised observation vector.

        Layout (all values normalised to [0, 1] unless noted):
          [0]   player_x / (SCREEN_WIDTH - PLAYER_WIDTH)
          [1]   shoot_cooldown / BULLET_COOLDOWN
          [2]   lives / PLAYER_START_LIVES
          For each of the OBS_MAX_ENEMIES enemy slots:
            [3 + i*3]     enemy_x / SCREEN_WIDTH   (or -1 if slot empty)
            [3 + i*3 + 1] enemy_y / SCREEN_HEIGHT  (or -1 if slot empty)
            [3 + i*3 + 2] 1.0 if active else -1.0
          For each of the OBS_MAX_BULLETS bullet slots:
            [3 + OBS_MAX_ENEMIES*3 + j*2]     bullet_x / SCREEN_WIDTH
            [3 + OBS_MAX_ENEMIES*3 + j*2 + 1] bullet_y / SCREEN_HEIGHT
        """
        obs: List[float] = []

        # Player state
        obs.append(self.player.x / max(1, SCREEN_WIDTH - PLAYER_WIDTH))
        obs.append(self.player.shoot_cooldown / BULLET_COOLDOWN)
        obs.append(self.player.lives / PLAYER_START_LIVES)

        # Enemies (sorted by proximity to player to make the closest most salient)
        enemies_sorted = sorted(
            self.enemies,
            key=lambda e: abs(e.center_x() - self.player.center_x()),
        )
        for i in range(OBS_MAX_ENEMIES):
            if i < len(enemies_sorted):
                e = enemies_sorted[i]
                obs.append(e.x / SCREEN_WIDTH)
                obs.append(e.y / SCREEN_HEIGHT)
                obs.append(1.0)
            else:
                obs.extend([-1.0, -1.0, -1.0])

        # Bullets
        bullets_sorted = sorted(self.bullets, key=lambda b: b.y)
        for j in range(OBS_MAX_BULLETS):
            if j < len(bullets_sorted):
                b = bullets_sorted[j]
                obs.append(b.x / SCREEN_WIDTH)
                obs.append(b.y / SCREEN_HEIGHT)
            else:
                obs.extend([-1.0, -1.0])

        return obs

    @property
    def observation_size(self) -> int:
        """Length of the observation vector."""
        return 3 + OBS_MAX_ENEMIES * 3 + OBS_MAX_BULLETS * 2

    @property
    def n_actions(self) -> int:
        """Number of discrete actions."""
        return len(Action)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fire_bullet(self) -> None:
        self.bullets.append(
            Bullet(
                x=self.player.center_x() - BULLET_WIDTH / 2,
                y=self.player.y - BULLET_HEIGHT,
            )
        )

    def _spawn_enemy(self) -> None:
        x = self._rng.randint(0, SCREEN_WIDTH - ENEMY_WIDTH)
        self.enemies.append(Enemy(x=float(x), y=float(-ENEMY_HEIGHT)))
