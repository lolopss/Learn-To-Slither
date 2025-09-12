# Python
# lobby.py
import pygame
import numpy as np
import IA_Snake as IA

TITLE = "Learn2Slither"
LOBBY_W = 480
LOBBY_H = 280


def run_lobby(save_path="./learning_state/q_table.npy", episodes=1001,
              visual="on"):
    """Small lobby: title + vertical buttons (Play, Train, Quit)."""
    def _load_q(path):
        try:
            return np.load(path)
        except Exception:
            return np.zeros((IA.state_space_size, IA.action_space_size),
                            dtype=float)

    def _make_screen():
        pygame.init()
        screen = pygame.display.set_mode((LOBBY_W, LOBBY_H))
        pygame.display.set_caption(TITLE)
        return screen

    def _draw(screen):
        screen.fill((20, 24, 40))
        font_big = pygame.font.SysFont(None, 42)
        font = pygame.font.SysFont(None, 26)

        # Title centered
        title_surf = font_big.render(TITLE, True, (220, 240, 255))
        title_rect = title_surf.get_rect(center=(LOBBY_W // 2, 36))
        screen.blit(title_surf, title_rect)

        # Vertical buttons centered under title
        btn_w, btn_h, gap = 260, 52, 14
        x = (LOBBY_W - btn_w) // 2
        y0 = title_rect.bottom + 16

        btns = {
            "play":  pygame.Rect(x, y0 + 0*(btn_h + gap), btn_w, btn_h),
            "train": pygame.Rect(x, y0 + 1*(btn_h + gap), btn_w, btn_h),
            "quit":  pygame.Rect(x, y0 + 2*(btn_h + gap), btn_w, btn_h),
        }

        labels = {
            "play": "Play 1 Game (P)",
            "train": "Train (T)",
            "quit": "Quit (Q/ESC)",
        }
        colors = {
            "play": (70, 130, 180),
            "train": (46, 139, 87),
            "quit": (200, 80, 80),
        }

        for key, rect in btns.items():
            pygame.draw.rect(screen, colors[key], rect, border_radius=8)
            pygame.draw.rect(screen, (255, 255, 255), rect, 2, border_radius=8)
            t = font.render(labels[key], True, (0, 0, 0))
            screen.blit(t, (rect.x + (btn_w - t.get_width()) // 2,
                            rect.y + (btn_h - t.get_height()) // 2))

        pygame.display.flip()
        return btns

    screen = _make_screen()
    btns = _draw(screen)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.quit()
                    return
                elif event.key == pygame.K_p:
                    pygame.quit()
                    IA.play_single_game(save_path, _load_q(save_path))
                    screen = _make_screen()
                    btns = _draw(screen)
                elif event.key == pygame.K_t:
                    pygame.quit()
                    IA.train_agent(save_path, episodes, visual)
                    screen = _make_screen()
                    btns = _draw(screen)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = pygame.mouse.get_pos()
                if btns["quit"].collidepoint(mx, my):
                    pygame.quit()
                    return
                if btns["play"].collidepoint(mx, my):
                    pygame.quit()
                    IA.play_single_game(save_path, _load_q(save_path))
                    screen = _make_screen()
                    btns = _draw(screen)
                if btns["train"].collidepoint(mx, my):
                    pygame.quit()
                    IA.train_agent(save_path, episodes, visual)
                    screen = _make_screen()
                    btns = _draw(screen)


if __name__ == "__main__":
    run_lobby()
