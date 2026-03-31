#!/usr/bin/env python3
"""Entry point for playing Space Shooter interactively.

Run with::

    python main.py

Controls
--------
Arrow Left / A  – move left
Arrow Right / D – move right
Space           – shoot
Q / Escape      – quit
R               – restart after game over
"""

from __future__ import annotations

import sys

import pygame

from space_shooter.game import Action, SpaceShooterGame
from space_shooter.renderer import PygameRenderer


def main() -> None:
    game = SpaceShooterGame()
    renderer = PygameRenderer()

    running = True
    while running:
        # ── Handle events ──────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r and game.done:
                    game.reset()

        # ── Map keyboard to action ──────────────────────────────────────
        action = Action.NOOP
        if not game.done:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                action = Action.LEFT
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                action = Action.RIGHT
            if keys[pygame.K_SPACE]:
                action = Action.SHOOT

        # ── Step game ──────────────────────────────────────────────────
        if not game.done:
            game.step(int(action))

        # ── Render ─────────────────────────────────────────────────────
        renderer.render(game, "human")

    renderer.close()
    sys.exit(0)


if __name__ == "__main__":
    main()
