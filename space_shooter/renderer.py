"""Pygame renderer for the Space Shooter game.

This module is intentionally isolated from the core game logic so that the
game can run headlessly (e.g., during RL training) without importing pygame.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .game import (
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    SpaceShooterGame,
)

# ---------------------------------------------------------------------------
# Colours (RGB)
# ---------------------------------------------------------------------------

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
DARK_BLUE = (5, 5, 30)
CYAN = (0, 220, 255)
RED = (220, 50, 50)
YELLOW = (255, 215, 0)
GREEN = (50, 200, 50)
GREY = (150, 150, 150)
ORANGE = (255, 140, 0)

# Star field
_N_STARS = 80


class PygameRenderer:
    """Handles all pygame rendering for the Space Shooter game."""

    def __init__(self) -> None:
        import pygame
        pygame.init()
        self._pygame = pygame
        self._screen: Optional[pygame.Surface] = None
        self._clock = pygame.time.Clock()
        self._font_large: Optional[pygame.font.Font] = None
        self._font_small: Optional[pygame.font.Font] = None
        self._stars: list = []

    def render(self, game: SpaceShooterGame, mode: str) -> Optional[np.ndarray]:
        """Render the game state.

        Args:
            game: The game instance to render.
            mode: ``"human"`` or ``"rgb_array"``.

        Returns:
            An ``(H, W, 3)`` uint8 array for ``mode="rgb_array"``, else ``None``.
        """
        pygame = self._pygame
        surface = self._get_surface(mode)
        if not self._stars:
            import random
            rng = random.Random(42)
            self._stars = [
                (rng.randint(0, SCREEN_WIDTH), rng.randint(0, SCREEN_HEIGHT))
                for _ in range(_N_STARS)
            ]

        if self._font_large is None:
            self._font_large = pygame.font.SysFont("monospace", 22, bold=True)
            self._font_small = pygame.font.SysFont("monospace", 16)

        # Background
        surface.fill(DARK_BLUE)
        for sx, sy in self._stars:
            pygame.draw.circle(surface, GREY, (sx, sy), 1)

        # Enemies
        for enemy in game.enemies:
            ex, ey, ew, eh = int(enemy.x), int(enemy.y), enemy.width, enemy.height
            # Body
            pygame.draw.polygon(
                surface,
                RED,
                [
                    (ex + ew // 2, ey),
                    (ex, ey + eh),
                    (ex + ew, ey + eh),
                ],
            )
            # Engine glow
            pygame.draw.circle(
                surface, ORANGE, (ex + ew // 2, ey + eh), 5
            )

        # Bullets
        for bullet in game.bullets:
            bx, by, bw, bh = int(bullet.x), int(bullet.y), bullet.width, bullet.height
            pygame.draw.rect(surface, YELLOW, (bx, by, bw, bh), border_radius=3)

        # Player ship
        px, py = int(game.player.x), int(game.player.y)
        pw, ph = game.player.width, game.player.height
        # Main body
        pygame.draw.polygon(
            surface,
            CYAN,
            [
                (px + pw // 2, py),
                (px, py + ph),
                (px + pw, py + ph),
            ],
        )
        # Engine highlight
        pygame.draw.circle(surface, WHITE, (px + pw // 2, py + ph), 4)

        # HUD
        score_surf = self._font_large.render(
            f"Score: {game.score}", True, WHITE
        )
        surface.blit(score_surf, (10, 10))

        lives_surf = self._font_large.render(
            f"Lives: {game.player.lives}", True, GREEN
        )
        surface.blit(lives_surf, (SCREEN_WIDTH - 120, 10))

        level_surf = self._font_small.render(
            f"Level {game._level}  Frame {game.frame}", True, GREY
        )
        surface.blit(level_surf, (10, 38))

        # Game-over overlay
        if game.done:
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 160))
            surface.blit(overlay, (0, 0))
            go_surf = self._font_large.render("GAME OVER", True, RED)
            sc_surf = self._font_small.render(
                f"Final Score: {game.score}", True, WHITE
            )
            surface.blit(
                go_surf,
                (SCREEN_WIDTH // 2 - go_surf.get_width() // 2, SCREEN_HEIGHT // 2 - 30),
            )
            surface.blit(
                sc_surf,
                (SCREEN_WIDTH // 2 - sc_surf.get_width() // 2, SCREEN_HEIGHT // 2 + 10),
            )

        if mode == "human":
            pygame.display.flip()
            self._clock.tick(60)
            return None
        else:  # rgb_array
            return np.transpose(
                pygame.surfarray.array3d(surface), axes=(1, 0, 2)
            )

    def close(self) -> None:
        self._pygame.quit()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_surface(self, mode: str):
        pygame = self._pygame
        if mode == "human":
            if self._screen is None:
                self._screen = pygame.display.set_mode(
                    (SCREEN_WIDTH, SCREEN_HEIGHT)
                )
                pygame.display.set_caption("Space Shooter — RL Game")
            return self._screen
        else:
            # Off-screen surface for rgb_array
            return pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
