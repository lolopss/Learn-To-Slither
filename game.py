#!/usr/bin/env python3
import pygame
import random
from IA_visualisation import draw_right_panel

# ---------------- CONFIG ---------------- #
BOARD_SIZE = 10    # grid is BOARD_SIZE x BOARD_SIZE
CELL_SIZE = 40       # pixel size of each cell
INFO_PANEL_WIDTH = 600   # space on the right for smoother interface
SIDE_PANEL_WIDTH = 100   # space on the left for additional information
TOP_PANEL_HEIGHT = 50    # space on top for time + score
BOTTOM_PANEL_HEIGHT = 50  # space on bottom
FPS = 3              # frames per second (snake speed)

WINDOW_WIDTH = BOARD_SIZE * CELL_SIZE + INFO_PANEL_WIDTH + SIDE_PANEL_WIDTH
WINDOW_HEIGHT = BOARD_SIZE * CELL_SIZE + TOP_PANEL_HEIGHT + BOTTOM_PANEL_HEIGHT


# ---------------- GAME CLASS ---------------- #
class SnakeGame:
    def __init__(self, board_size=BOARD_SIZE):
        self.board_size = board_size
        self.ate_green = 0
        self.ate_red = 0
        self.previous_distance = BOARD_SIZE // 2
        self.reset()

    def reset(self):
        # Snake starts length 3, horizontal, in the middle of the board
        start_x = self.board_size // 2
        start_y = self.board_size // 2
        self.snake = [(start_x - i, start_y) for i in range(3)]  # Horizontal
        self.direction = (1, 0)  # Moving right
        self.spawn_apples()
        self.alive = True
        self.direction_changed = False  # Track change in direction for step

    def spawn_apples(self):
        """Always 2 green apples and 1 red apple on the board."""
        occupied = set(self.snake)  # Include the snake's body as occupied
        self.green_apples = []
        self.red_apples = []

        # Generate all possible positions
        all_positions = [
            (x, y) for x in range(self.board_size) for y
            in range(self.board_size)
        ]
        available_positions = [
            pos for pos in all_positions if pos not in occupied
        ]

        # Check if we have enough space
        if len(available_positions) < 3:  # 2 green + 1 red
            print("Warning: Not enough space for apples!")
            return

        # Randomly select positions for apples
        selected_positions = random.sample(available_positions, 3)

        # Assign first 2 to green apples, last 1 to red apple
        self.green_apples = selected_positions[:2]
        self.red_apples = [selected_positions[2]]

    def step(self):
        if not self.alive:
            return

        self.direction_changed = False  # Allow direction change for the step

        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)

        # Wall collision
        if new_head[0] < 0 or new_head[0] >= self.board_size \
           or new_head[1] < 0 or new_head[1] >= self.board_size:
            self.alive = False
            return

        # Self collision
        if new_head in self.snake:
            self.alive = False
            return

        # Eat green apple (+1)
        if new_head in self.green_apples:
            self.ate_green = 1
            self.ate_red = 0
            self.green_apples.remove(new_head)
            self.snake.insert(0, new_head)  # Add head, don't remove tail
            self.add_apple(self.green_apples)  # Add a new green apple
            return

        # Eat red apple (-1)
        if new_head in self.red_apples:
            self.ate_red = 1  # Set ate_red to 1
            self.ate_green = 0  # Reset ate_green
            self.red_apples.remove(new_head)
            self.snake.insert(0, new_head)  # Add new head first

            # Remove tail (normal move)
            if len(self.snake) > 1:
                self.snake.pop()

            # Remove one more segment (shrink effect)
            if len(self.snake) > 1:
                self.snake.pop()

            # Check if snake is too small
            if len(self.snake) < 1:
                self.alive = False
                return

            self.add_apple(self.red_apples)  # Add a new red apple
            return

        # Normal move
        self.ate_green = 0  # Reset ate_green
        self.ate_red = 0    # Reset ate_red
        self.snake.insert(0, new_head)
        self.snake.pop()

    def add_apple(self, apple_list):
        """Add a single apple to the specified list (green or red)."""
        occupied = set(self.snake + self.green_apples + self.red_apples)
        while True:
            pos = (random.randrange(self.board_size),
                   random.randrange(self.board_size))
            if pos not in occupied:
                apple_list.append(pos)
                break

    def change_direction(self, new_dir):
        if self.direction_changed:
            return  # Ignore further direction changes in the same step

        dx, dy = self.direction
        ndx, ndy = new_dir
        # Prevent 180° turns (turning back on itself)
        if (dx + ndx, dy + ndy) != (0, 0):  # Opposite directions sum to (0, 0)
            self.direction = new_dir
            self.direction_changed = True  # Mark direction as changed

    def get_board(self):
        """ Generate a 2D grid representation
            of the game with walls
            outside the playable area."""
        # Create an empty board filled with '0' (empty cells), including walls
        board = [['W' if x == 0 or x == self.board_size + 1 or y == 0
                  or y == self.board_size + 1 else '0'
                  for x in range(self.board_size + 2)] for y in range(
                      self.board_size + 2)]

        # Add the snake's body
        for i, (x, y) in enumerate(self.snake):
            if i == 0:
                board[y + 1][x + 1] = 'H'  # Head of the snake
            else:
                board[y + 1][x + 1] = 'S'  # Body of the snake

        # Add green apples
        for x, y in self.green_apples:
            board[y + 1][x + 1] = 'G'

        # Add red apples
        for x, y in self.red_apples:
            board[y + 1][x + 1] = 'R'

        return board


# ---------------- DRAWING ---------------- #

# Used to hange visibility in game (V BUTTON)
def _compute_visible_cells(game):
    size = game.board_size
    head_x, head_y = game.snake[0]
    vis = {(head_x, head_y)}
    for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
        x, y = head_x, head_y
        while True:
            x += dx
            y += dy
            if x < 0 or x >= size or y < 0 or y >= size:
                break
            vis.add((x, y))
    return vis


def compute_direction_scan(game):
    """
    Returns per-direction info for the 4 straight rays from the head:
      directions order: UP(0), RIGHT(1), DOWN(2), LEFT(3)

    For each direction we compute:
        immediate_block  : 1 if the NEXT cell is wall or snake body
        green_visible    : 1 if any green apple is somewhere along that ray
        red_visible      : 1 if any red apple is along the ray
        body_visible     : 1 if any (non-head) snake segment is along the ray
                           (past the first cell)
        free_len         : how many free cells (not wall, not body) until
                           collision/wall
    Returns:
        dict with keys:
            'headers' : ["UP","RIGHT","DOWN","LEFT"]
            'rows'    : list of row descriptors:
                [
                  ("ImmBlk", [..4 ints..]),
                  ("Green" , [..]),
                  ("Red"   , [..]),
                  ("Body"  , [..]),
                  ("Free"  , [..]),
                  ("Total" , [..])  # Simple sum of binary lines except Free
                ]
    """
    head_x, head_y = game.snake[0]
    body = set(game.snake[1:])
    greens = set(game.green_apples)
    reds = set(game.red_apples)

    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    headers = ["UP", "RIGHT", "DOWN", "LEFT"]

    imm = []
    gvis = []
    rvis = []
    bvis = []
    free = []

    size = game.board_size

    for dx, dy in directions:
        nx, ny = head_x + dx, head_y + dy
        # immediate block
        immediate_block = int(nx < 0 or nx >= size or ny < 0 or ny >= size or (
            nx, ny) in body)
        imm.append(immediate_block)

        # scan ray
        green_seen = 0
        red_seen = 0
        body_seen = 0
        free_cells = 0

        cx, cy = head_x, head_y
        while True:
            cx += dx
            cy += dy
            if cx < 0 or cx >= size or cy < 0 or cy >= size:
                break
            pos = (cx, cy)
            if pos in body:
                body_seen = 1
                break
            if pos in greens:
                green_seen = 1
            if pos in reds:
                red_seen = 1
            free_cells += 1
            # if both apples seen we can still continue for body but
            # break early if desired
            # keep scanning to know if body occurs behind apples:
            # (optional early-break omitted for clarity)

        gvis.append(green_seen)
        rvis.append(red_seen)
        bvis.append(body_seen)
        free.append(free_cells)

    # total = sum of selected binary rows (choose which to aggregate)
    total = [imm[i] + gvis[i] + rvis[i] + bvis[i] for i in range(4)]

    rows = [
        ("ImmBlk", imm),
        ("Green", gvis),
        ("Red", rvis),
        ("Body", bvis),
        ("Free", free),
        ("Total", total),
    ]
    return {
        "headers": headers,
        "rows": rows
    }


def draw_game(screen, game: 'SnakeGame', elapsed_time, total_reward=0,
              last_action=None, action_values=None, show_info=True,
              next_is_explore=False, vision_only=False):
    screen.fill((0, 0, 90))
    board_surface = pygame.Surface((BOARD_SIZE * CELL_SIZE, BOARD_SIZE *
                                    CELL_SIZE))
    board_surface.fill((50, 50, 50))

    # Precompute visibility if enabled
    visible = _compute_visible_cells(
        game) if vision_only and game.snake else None

    # grid
    for x in range(game.board_size):
        for y in range(game.board_size):
            pygame.draw.rect(board_surface, (70, 70, 70),
                             pygame.Rect(x*CELL_SIZE,
                                         y*CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)

    # snake
    for i, (x, y) in enumerate(game.snake):
        if visible is not None and (x, y) not in visible:
            continue
        rect = pygame.Rect(x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
        if i == 0:
            pygame.draw.rect(board_surface, (0, 200, 150), rect)
            eye_size = CELL_SIZE // 5
            off_x = CELL_SIZE // 4
            off_y = CELL_SIZE // 4
            pygame.draw.rect(board_surface, (255, 255, 255),
                             pygame.Rect(x*CELL_SIZE+off_x, y*CELL_SIZE+off_y,
                                         eye_size, eye_size))
            pygame.draw.rect(board_surface, (255, 255, 255),
                             pygame.Rect(x*CELL_SIZE+CELL_SIZE-off_x-eye_size,
                                         y*CELL_SIZE+off_y, eye_size,
                                         eye_size))
        else:
            pygame.draw.rect(board_surface, (0, 150, 100), rect)

    # apples
    for (x, y) in game.green_apples:
        if visible is not None and (x, y) not in visible:
            continue
        pygame.draw.rect(board_surface, (100, 255, 100),
                         pygame.Rect(x*CELL_SIZE, y*CELL_SIZE,
                                     CELL_SIZE, CELL_SIZE))
    for (x, y) in game.red_apples:
        if visible is not None and (x, y) not in visible:
            continue
        pygame.draw.rect(board_surface, (255, 100, 100),
                         pygame.Rect(x*CELL_SIZE, y*CELL_SIZE,
                                     CELL_SIZE, CELL_SIZE))

    screen.blit(board_surface, (SIDE_PANEL_WIDTH, TOP_PANEL_HEIGHT))
    pygame.draw.rect(screen, (0, 0, 139),
                     pygame.Rect(0, 0, WINDOW_WIDTH, TOP_PANEL_HEIGHT))
    font = pygame.font.SysFont(None, 36)
    snake_size_text = font.render(f"Snake Size: {len(game.snake)}\
                                  ", True, (255, 255, 255))
    screen.blit(snake_size_text,
                snake_size_text.get_rect(center=(WINDOW_WIDTH//2.9,
                                                 TOP_PANEL_HEIGHT//2)))
    elapsed_time_text = font.render(f"Time: {int(elapsed_time)}s", True,
                                    (255, 255, 255))
    screen.blit(elapsed_time_text,
                elapsed_time_text.get_rect(center=(WINDOW_WIDTH//6,
                                                   TOP_PANEL_HEIGHT // 2)))

    pygame.draw.rect(screen, (0, 0, 139),
                     pygame.Rect(0, WINDOW_HEIGHT-BOTTOM_PANEL_HEIGHT,
                                 WINDOW_WIDTH, BOTTOM_PANEL_HEIGHT))
    pygame.draw.rect(screen, (0, 0, 139),
                     pygame.Rect(0, 0, SIDE_PANEL_WIDTH, WINDOW_HEIGHT))
    # Right panel call
    if show_info:
        draw_right_panel(screen, game,
                         elapsed_time=elapsed_time,
                         total_reward=total_reward,
                         last_action=last_action,
                         action_values=action_values,
                         info_panel_width=INFO_PANEL_WIDTH,
                         next_is_explore=next_is_explore)

    pygame.display.flip()

# if __name__ == "__main__":
#     game_loop()
#     print()
