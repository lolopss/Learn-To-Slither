#!/usr/bin/env python3
import pygame
import random
import sys

# ---------------- CONFIG ---------------- #
BOARD_SIZE = 10      # grid is BOARD_SIZE x BOARD_SIZE
CELL_SIZE = 40       # pixel size of each cell
INFO_PANEL_WIDTH = 300   # space on the right for additional information
SIDE_PANEL_WIDTH = 100   # space on the left for additional information
TOP_PANEL_HEIGHT = 50    # space on top for instructions/score
BOTTOM_PANEL_HEIGHT = 50 # space on bottom for additional information
FPS = 15              # frames per second (snake speed)

WINDOW_WIDTH = BOARD_SIZE * CELL_SIZE + INFO_PANEL_WIDTH + SIDE_PANEL_WIDTH
WINDOW_HEIGHT = BOARD_SIZE * CELL_SIZE + TOP_PANEL_HEIGHT + BOTTOM_PANEL_HEIGHT


# ---------------- GAME CLASS ---------------- #
class SnakeGame:
    def __init__(self, board_size=BOARD_SIZE):
        self.board_size = board_size
        self.reset()

    def reset(self):
        # Snake starts length 3, horizontal, in the middle of the board
        start_x = self.board_size // 2
        start_y = self.board_size // 2
        self.snake = [(start_x - i, start_y) for i in range(3)]  # Horizontal snake
        self.direction = (1, 0)  # Moving right
        self.spawn_apples()
        self.alive = True
        self.direction_changed = False  # Track if direction was changed in this step

    def spawn_apples(self):
        """Always 2 green apples and 1 red apple on the board."""
        occupied = set(self.snake)
        self.green_apples = []
        self.red_apples = []

        while len(self.green_apples) < 2:
            pos = (random.randrange(self.board_size),
                   random.randrange(self.board_size))
            if pos not in occupied:
                self.green_apples.append(pos)

        while len(self.red_apples) < 1:
            pos = (random.randrange(self.board_size),
                    random.randrange(self.board_size))
            if pos not in occupied and pos not in self.green_apples:
                self.red_apples.append(pos)

    def step(self):
        if not self.alive:
            return

        self.direction_changed = False  # Allow direction change for the new step

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

        self.snake.insert(0, new_head)

        # Eat green apple (+1)
        if new_head in self.green_apples:
            self.green_apples.remove(new_head)
            self.add_apple(self.green_apples)  # Add a new green apple
            return

        # Eat red apple (-1)
        if new_head in self.red_apples:
            self.red_apples.remove(new_head)
            self.snake.pop()  # Normal move
            if self.snake:
                self.snake.pop()  # Shrink extra
            self.add_apple(self.red_apples)  # Add a new red apple
            if not self.snake:
                self.alive = False
            return

        # Normal move
        self.snake.pop()

    def add_apple(self, apple_list):
        """Add a single apple to the specified list (green or red)."""
        occupied = set(self.snake + self.green_apples + self.red_apples)
        while True:
            pos = (random.randrange(self.board_size), random.randrange(self.board_size))
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


# ---------------- DRAWING ---------------- #
def draw_game(screen, game: SnakeGame):
    # Fill the entire screen with dark blue
    screen.fill((0, 0, 90))  # Dark blue background for all panels

    # Game board area
    board_surface = pygame.Surface(
        (BOARD_SIZE * CELL_SIZE, BOARD_SIZE * CELL_SIZE))
    board_surface.fill((50, 50, 50))  # Dark gray background for the board

    # Grid
    for x in range(game.board_size):
        for y in range(game.board_size):
            rect = pygame.Rect(
                x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(board_surface, (70, 70, 70), rect, 1)  # Light gray grid lines

    # Snake
    for i, (x, y) in enumerate(game.snake):
        rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        if i == 0:  # Head of the snake
            pygame.draw.rect(board_surface, (0, 200, 150), rect)  # Teal color for the head

            # Draw eyes on the head
            eye_size = CELL_SIZE // 5
            eye_offset_x = CELL_SIZE // 4
            eye_offset_y = CELL_SIZE // 4

            # Left eye
            left_eye = pygame.Rect(
                x * CELL_SIZE + eye_offset_x,
                y * CELL_SIZE + eye_offset_y,
                eye_size,
                eye_size
            )
            pygame.draw.rect(board_surface, (255, 255, 255), left_eye)

            # Right eye
            right_eye = pygame.Rect(
                x * CELL_SIZE + CELL_SIZE - eye_offset_x - eye_size,
                y * CELL_SIZE + eye_offset_y,
                eye_size,
                eye_size
            )
            pygame.draw.rect(board_surface, (255, 255, 255), right_eye)
        else:
            pygame.draw.rect(board_surface, (0, 150, 100), rect)  # Greenish color for the body

    # Green apples
    for (x, y) in game.green_apples:
        rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(board_surface, (100, 255, 100), rect)  # Bright green apples

    # Red apples
    for (x, y) in game.red_apples:
        rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(board_surface, (255, 100, 100), rect)  # Soft red apples

    # Blit the game board onto the screen at the correct position
    screen.blit(board_surface, (SIDE_PANEL_WIDTH, TOP_PANEL_HEIGHT))

    # Draw additional panels
    # Top panel
    pygame.draw.rect(screen, (0, 0, 139), (0, 0, WINDOW_WIDTH, TOP_PANEL_HEIGHT))  # Dark blue top panel

    # Display snake size in the middle of the top panel
    font = pygame.font.SysFont(None, 36)  # Use a default font with size 36
    snake_size_text = font.render(f"Snake Size: {len(game.snake)}", True, (255, 255, 255))  # White text
    text_rect = snake_size_text.get_rect(center=(WINDOW_WIDTH // 2, TOP_PANEL_HEIGHT // 2))  # Centered in the top panel
    screen.blit(snake_size_text, text_rect)

    # Bottom panel
    pygame.draw.rect(screen, (0, 0, 139), (0, WINDOW_HEIGHT - BOTTOM_PANEL_HEIGHT, WINDOW_WIDTH, BOTTOM_PANEL_HEIGHT))  # Dark blue bottom panel

    # Left panel
    pygame.draw.rect(screen, (0, 0, 139), (0, 0, SIDE_PANEL_WIDTH, WINDOW_HEIGHT))  # Dark blue left panel

    # Right panel
    pygame.draw.rect(screen, (0, 0, 139), (WINDOW_WIDTH - INFO_PANEL_WIDTH, 0, INFO_PANEL_WIDTH, WINDOW_HEIGHT))  # Dark blue right panel

    pygame.display.flip()


# ---------------- MAIN LOOP ---------------- #
def game_loop():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Learn2Slither Base Game")
    clock = pygame.time.Clock()

    game = SnakeGame(BOARD_SIZE)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    game.change_direction((0, -1))
                elif event.key == pygame.K_DOWN:
                    game.change_direction((0, 1))
                elif event.key == pygame.K_LEFT:
                    game.change_direction((-1, 0))
                elif event.key == pygame.K_RIGHT:
                    game.change_direction((1, 0))

        if game.alive:
            game.step()

        draw_game(screen, game)
        clock.tick(FPS) # Speed of the snake


if __name__ == "__main__":
    game_loop()
