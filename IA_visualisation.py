import pygame

# Fallback defaults (overridden by passed args from game.draw_game)
DEFAULT_INFO_PANEL_WIDTH = 600

COLORS = {
    "BG": (0, 0, 90),
    "PANEL_BG": (10, 10, 40),
    "BORDER": (80, 80, 140),
    "TEXT": (230, 230, 230),
    "HEADERS": (180, 200, 255),
    "WALL": (255, 200, 50),
    "GREEN": (80, 230, 80),
    "RED": (240, 90, 90),
    "BODY": (180, 120, 230),
    "FREE": (100, 170, 240),
    "TOTAL": (255, 215, 0),
}

DIRECTION_LABELS = ["UP", "RIGHT", "DOWN", "LEFT"]
CATEGORY_ROWS = [
    ("ImmBlk", "WALL"),
    ("Green", "GREEN"),
    ("Red", "RED"),
    ("Body", "BODY"),
    ("Free", "FREE"),
    ("Total", "TOTAL"),
]


def compute_direction_scan(game):
    head_x, head_y = game.snake[0]
    body = set(game.snake[1:])
    greens = set(game.green_apples)
    reds = set(game.red_apples)
    directions = [(0,-1),(1,0),(0,1),(-1,0)]
    imm,gvis,rvis,bvis,free = [],[],[],[],[]
    size = game.board_size
    for dx, dy in directions:
        nx, ny = head_x + dx, head_y + dy
        imm.append(int(nx < 0 or nx >= size or ny < 0 or ny >= size or (nx, ny) in body))
        green_seen = 0; red_seen = 0; body_seen = 0; free_cells = 0
        cx, cy = head_x, head_y
        while True:
            cx += dx; cy += dy
            if cx < 0 or cx >= size or cy < 0 or cy >= size:
                break
            pos = (cx, cy)
            if pos in body:
                body_seen = 1
                break
            if pos in greens: green_seen = 1
            if pos in reds:   red_seen = 1
            free_cells += 1
        gvis.append(green_seen); rvis.append(red_seen)
        bvis.append(body_seen);  free.append(free_cells)
    total = [imm[i]+gvis[i]+rvis[i]+bvis[i] for i in range(4)]
    rows = [("ImmBlk", imm), ("Green", gvis), ("Red", rvis),
            ("Body", bvis), ("Free", free), ("Total", total)]
    return {"headers": ["UP","RIGHT","DOWN","LEFT"], "rows": rows}

def _circle_color(row_key, base_rgb, value, max_free=None):
    if row_key == "Free" and max_free:
        ratio = 0 if max_free == 0 else min(1.0, value / max_free)
        scale = 0.25 + 0.75 * ratio
    else:
        scale = 0.25 if value == 0 else 0.55 + 0.45 * min(1, value)
    return tuple(min(255, int(c * scale)) for c in base_rgb)

def draw_state_panel(screen, game, info, panel_x, panel_y,
                     last_action=None, action_values=None,
                     info_panel_width=DEFAULT_INFO_PANEL_WIDTH):
    font_dir = pygame.font.SysFont(None, 18)
    font_cat = pygame.font.SysFont(None, 18)
    font_val = pygame.font.SysFont(None, 16)
    font_title = pygame.font.SysFont(None, 22)
    width = info_panel_width - 20
    header_h = 28; row_h = 40; title_h = 34
    total_h = title_h + header_h + len(CATEGORY_ROWS)*row_h + 14
    panel_rect = pygame.Rect(panel_x, panel_y, width, total_h)
    pygame.draw.rect(screen, COLORS["PANEL_BG"], panel_rect, border_radius=8)
    pygame.draw.rect(screen, COLORS["BORDER"], panel_rect, 2, border_radius=8)
    title = font_title.render("SNAKE VISION", True, COLORS["HEADERS"])
    screen.blit(title, (panel_x + (width - title.get_width()) // 2, panel_y + 6))
    grid_x = panel_x + 8
    grid_y = panel_y + title_h
    col_w = (width - 80) // 4
    for d_i, label in enumerate(DIRECTION_LABELS):
        hx = grid_x + 80 + d_i * col_w
        hdr_col = (70,130,180) if d_i == 0 else (205,133,63) if d_i == 1 else (138,43,226) if d_i == 2 else (46,139,87)
        if last_action == d_i:
            hdr_col = tuple(min(255, int(c * 1.6)) for c in hdr_col)
        pygame.draw.rect(screen, hdr_col, (hx, grid_y, col_w - 6, header_h - 6), border_radius=4)
        txt = font_dir.render(label, True, (0,0,0))
        screen.blit(txt, (hx + (col_w - 6 - txt.get_width())//2,
                          grid_y + (header_h - 6 - txt.get_height())//2))
    max_free = max(info["rows"][4][1]) if info["rows"][4][0] == "Free" else 1
    for r_i, (row_key, values) in enumerate(info["rows"]):
        ry = grid_y + header_h + r_i * row_h
        color_name = next(cn for (rk, cn) in CATEGORY_ROWS if rk == row_key)
        base_color = COLORS[color_name]
        lbl = font_cat.render(row_key, True, base_color)
        screen.blit(lbl, (grid_x + 8, ry + (row_h - lbl.get_height()) // 2))
        if row_key == "Free":
            best_idx = int(max(range(4), key=lambda i: values[i]))
        else:
            positives = [i for i,v in enumerate(values) if v > 0]
            best_idx = positives[0] if positives else int(max(range(4), key=lambda i: values[i]))
        for d_i, val in enumerate(values):
            cx = grid_x + 80 + d_i * col_w + (col_w // 2) - 4
            cy = ry + row_h // 2 - 6
            draw_col = _circle_color(row_key, base_color, val,
                                     max_free=max_free if row_key == "Free" else None)
            if d_i == best_idx and val > 0:
                draw_col = tuple(min(255, int(c * 1.8)) for c in base_color)
            pygame.draw.circle(screen, draw_col, (cx, cy), 14)
            vtxt = font_val.render(str(val), True, COLORS["TEXT"])
            screen.blit(vtxt, (cx - vtxt.get_width()//2, cy + 16))
    if action_values is not None:
        q_y = panel_y + total_h + 8
        q_rect = pygame.Rect(panel_x, q_y, width, 120)
        pygame.draw.rect(screen, COLORS["PANEL_BG"], q_rect, border_radius=8)
        pygame.draw.rect(screen, COLORS["BORDER"], q_rect, 2, border_radius=8)
        q_title = font_title.render("ACTIONS (Q)", True, COLORS["HEADERS"])
        screen.blit(q_title, (panel_x + (width - q_title.get_width()) // 2, q_y + 4))
        names = ["UP","DOWN","LEFT","RIGHT"]
        for i, (n, qv) in enumerate(zip(names, action_values)):
            bx = panel_x + 12 + i * ((width - 24)//4)
            by = q_y + 40
            act_col = (70,130,180) if i == 0 else (205,133,63) if i == 1 else (138,43,226) if i == 2 else (46,139,87)
            if last_action == i:
                act_col = tuple(min(255, int(c * 1.7)) for c in act_col)
            pygame.draw.circle(screen, act_col, (bx + 35, by), 22)
            at = font_val.render(n, True, (0,0,0))
            screen.blit(at, (bx + 35 - at.get_width()//2, by - at.get_height()//2))
            qtxt = font_val.render(f"{qv:.2f}", True, COLORS["TEXT"])
            screen.blit(qtxt, (bx + 35 - qtxt.get_width()//2, by + 26))

def draw_right_panel(screen, game, elapsed_time=0, total_reward=0,
                     last_action=None, action_values=None,
                     info_panel_width=DEFAULT_INFO_PANEL_WIDTH):
    info = compute_direction_scan(game)
    panel_x = screen.get_width() - info_panel_width + 10
    panel_y = 30
    # small stats top block
    font = pygame.font.SysFont(None, 20)
    stats = [
        f"Time: {int(elapsed_time)}s",
        f"Len: {len(game.snake)}",
        f"RewardΣ: {total_reward}",
    ]
    for i, line in enumerate(stats):
        txt = font.render(line, True, COLORS["TEXT"])
        screen.blit(txt, (panel_x, panel_y - 55 + i*18))
    draw_state_panel(screen, game, info, panel_x, panel_y,
                     last_action=last_action,
                     action_values=action_values,
                     info_panel_width=info_panel_width)