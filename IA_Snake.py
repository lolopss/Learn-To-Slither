import numpy as np
import random
import pygame
from game import SnakeGame, draw_game, WINDOW_WIDTH, WINDOW_HEIGHT

# Improved state space - much smaller and more focused
state_space_size = 8 ** 4           # 4 directions, each encoded on 3 bits
#                                     (danger, green, red)
action_space_size = 4     # Up, Down, Left, Right

Q_table = np.full((state_space_size, action_space_size), 3.0, dtype=float)

# Improved hyperparameters
alpha = 0.1         # learning rate
gamma = 0.9         # Higher discount factor - care more about future
epsilon = 0.1        # Exploration rate
epsilon_decay = 0.9995  # Slower decay
min_epsilon = 0.01   # Lower minimum exploration
explore_count = 0
exploit_count = 0

# To pick best game
interval_best_score = -1
interval_best_actions = []
interval_best_seed = None

# For graph.py
episode_lengths = []
# -------------------------------- STATE ENCODING ----------------------------- #


def encode_state(game):
    """
    Vision limited to 4 straight lines (UP, RIGHT, DOWN, LEFT).
    For each direction we collect:
        - danger_immediate (cell right next to head is wall or snake body)
        - any green apple somewhere further in that line (before wall)
        - any red apple somewhere further in that line
    No other info (no current heading, no relative apple direction, no diagonals).
    Each direction => 3 bits => a value 0..7:
        bit2 = danger_immediate
        bit1 = green_in_line
        bit0 = red_in_line
    State index = base-8 number formed by (UP, RIGHT, DOWN, LEFT) codes.
    """
    head_x, head_y = game.snake[0]
    directions = [(0,-1),(1,0),(0,1),(-1,0)]  # UP, RIGHT, DOWN, LEFT
    body = set(game.snake)
    greens = set(game.green_apples)
    reds = set(game.red_apples)

    codes = []
    for dx, dy in directions:
        nx, ny = head_x + dx, head_y + dy
        danger_immediate = (nx < 0 or nx >= game.board_size or
                            ny < 0 or ny >= game.board_size or
                            (nx, ny) in body)

        green_in_line = False
        red_in_line = False

        # Ray cast further along the direction until wall
        cx, cy = head_x, head_y
        while True:
            cx += dx
            cy += dy
            if cx < 0 or cx >= game.board_size or cy < 0 or cy >= game.board_size:
                break
            if (cx, cy) in greens:
                green_in_line = True
            if (cx, cy) in reds:
                red_in_line = True
            if green_in_line and red_in_line:
                break

        code = (danger_immediate << 2) | (green_in_line << 1) | red_in_line  # 0..7
        codes.append(code)

    state = 0
    for c in codes:          # base-8 accumulation
        state = state * 8 + c
    return state  # 0 .. 8^4 - 1


# --- (Optional) Safety: reallocate Q_table if loaded file has old shape ---
def ensure_qtable_shape():
    global Q_table
    if Q_table.shape != (state_space_size, action_space_size):
        Q_table = np.zeros((state_space_size, action_space_size))

# Call ensure_qtable_shape() right after loading a saved table in train_agent():
# try:
#     Q_table = np.load(save_path)
#     ensure_qtable_shape()
#     ...
# except FileNotFoundError:
#     ...
# ----------------------- REWARD FUNCTION ----------------- #

def get_reward(game):
    if not game.alive: return -120
    if game.ate_green: return 25
    if game.ate_red:   return -15

    head_x, head_y = game.snake[0]
    apple_x, apple_y = game.green_apples[0]
    dist = abs(head_x - apple_x) + abs(head_y - apple_y)

    if not hasattr(game, "recent_heads"):
        game.recent_heads = []
    game.recent_heads.append((head_x, head_y))
    if len(game.recent_heads) > 14: game.recent_heads.pop(0)

    closer = dist < game.previous_distance
    farther = dist > game.previous_distance
    game.previous_distance = dist

    reward = 0
    if closer: reward += 1.0
    elif farther: reward -= 1
    else: reward -= 0.5

    # loop penalty
    if len(game.recent_heads) == 14 and len(set(game.recent_heads)) <= 5:
        reward -= 3

    # mild step cost to encourage efficiency
    reward -= 0.05

    return reward

# -------------------------------- TRAINING LOOP ------------------------------- #


def get_valid_actions(game, allow_opposite=False):
    # 0=UP,1=DOWN,2=LEFT,3=RIGHT
    dirs = [(0,-1),(0,1),(-1,0),(1,0)]
    head_x, head_y = game.snake[0]
    cur_dx, cur_dy = game.direction
    size = game.board_size
    valid = []
    for a, (dx, dy) in enumerate(dirs):
        # avoid 180° turn (no-op due to change_direction rule)
        if not allow_opposite and (cur_dx + dx, cur_dy + dy) == (0, 0):
            continue
        nx, ny = head_x + dx, head_y + dy
        if nx < 0 or nx >= size or ny < 0 or ny >= size:
            continue
        if (nx, ny) in game.snake:
            continue
        valid.append(a)
    return valid

def safe_argmax(qrow, game):
    valid = get_valid_actions(game)
    if not valid:
        return random.randint(0, 3)
    # pick best among valid
    best_a = max(valid, key=lambda a: qrow[a])
    return int(best_a)

def choose_action(state, Q_table, epsilon, game):
    global explore_count, exploit_count

    # Safety: clamp/validate state
    if state < 0 or state >= Q_table.shape[0]:
        explore_count += 1
        valid = get_valid_actions(game)
        return random.choice(valid) if valid else random.randint(0, 3)

    row = Q_table[state]

    # Exploration
    if random.random() < epsilon:
        explore_count += 1
        valid = get_valid_actions(game)
        return random.choice(valid) if valid else random.randint(0, 3)

    # Exploitation (mask invalid)
    exploit_count += 1
    if not np.isfinite(row).all():
        explore_count += 1
        valid = get_valid_actions(game)
        return random.choice(valid) if valid else random.randint(0, 3)

    return safe_argmax(row, game)


def print_vision(game, step_idx=None):
    """
    Prints a BOARD_SIZE x BOARD_SIZE grid:
      '.' = not in any of the 4 vision rays (hidden)
      '0' = visible empty cell
      'H' = snake head (always visible)
      'S' = visible snake body segment
      'G' = visible green apple
      'R' = visible red apple
    Vision = straight rays UP / DOWN / LEFT / RIGHT from head until wall.
    """
    size = game.board_size
    head_x, head_y = game.snake[0]
    body = set(game.snake[1:])
    greens = set(game.green_apples)
    reds = set(game.red_apples)

    # Start with all hidden
    grid = [['.' for _ in range(size)] for _ in range(size)]

    def reveal(x, y):
        if (x, y) == (head_x, head_y):
            grid[y][x] = 'H'
        elif (x, y) in body:
            grid[y][x] = 'S'
        elif (x, y) in greens:
            grid[y][x] = 'G'
        elif (x, y) in reds:
            grid[y][x] = 'R'
        else:
            grid[y][x] = '0'

    # Head always visible
    reveal(head_x, head_y)

    # Rays: up, down, left, right
    for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
        x, y = head_x, head_y
        while True:
            x += dx
            y += dy
            if x < 0 or x >= size or y < 0 or y >= size:
                break
            reveal(x, y)

    prefix = f"[STEP {step_idx}] " if step_idx is not None else ""
    print(prefix)
    for row in grid:
        print(''.join(row))
    print()  # extra newline


def action_to_direction(action):
    """Convert action index to direction tuple."""
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right
    return directions[action]


def train_agent(save_path, nb_episodes, visual):
    global Q_table
    global epsilon
    global exploit_count
    global explore_count
    top_score = 0
    # Load Q-table if it exists
    try:
        Q_table = np.load(save_path)
        ensure_qtable_shape()
        print(f"Loaded Q-table from {save_path}")
    except FileNotFoundError:
        print(f"No Q-table found at {save_path}. Starting fresh.")

    # Track performance
    scores = []
    recent_scores = []
    recent_rewards = []  # Track total rewards for the last 100 episodes
    global interval_best_score, interval_best_actions, interval_best_seed
    interval_best_score = -1
    interval_best_actions = []
    interval_best_seed = None
    # Training loop
    for episode in range(nb_episodes):
        episode_actions = []
        episode_seed = random.randrange(1 << 30)
        random.seed(episode_seed)
        game = SnakeGame()
        game.reset()
        state = encode_state(game)
        total_reward = 0  # Track total reward for the current episode
        steps = 0

        while game.alive and steps < 1000:  # Prevent infinite loops
            # Choose action (exploration vs exploitation)
            action = choose_action(state, Q_table, epsilon, game)
            # Safety fallback
            if action is None or action not in (0, 1, 2, 3):
                action = random.randint(0, 3)
            episode_actions.append(action)
            # Convert action to direction and apply
            direction = action_to_direction(action)
            game.change_direction(direction)
            game.step()

            # Get reward before updating state (in case snake dies)
            reward = get_reward(game)
            next_state = encode_state(game)
            total_reward += reward
            steps += 1

            # Q-learning update
            best_next_action = np.max(Q_table[next_state]) if game.alive and game.snake else 0
            Q_table[state, action] = Q_table[state, action] + alpha * (
                reward + gamma * best_next_action - Q_table[state, action]
            )

            # Update for next iteration
            state = next_state

        # Record performance
        final_score = len(game.snake)
        episode_lengths.append(final_score)
        if final_score > interval_best_score:
            interval_best_score = final_score
            interval_best_actions = episode_actions[:]   # copy
            interval_best_seed = episode_seed
        scores.append(final_score)
        recent_scores.append(final_score)
        recent_rewards.append(total_reward)  # Add total reward for this episode

        if len(recent_scores) > 1000:
            recent_scores.pop(0)

        if len(recent_rewards) > 1000:
            recent_rewards.pop(0)

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        top_score = max(top_score, final_score)
        # Print progress
        if (episode + 1) % 1000 == 0:
            avg_score = np.mean(recent_scores) if recent_scores else 0
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0
            total = explore_count + exploit_count
            print(f"Episode {episode + 1}: Avg length (last 1000): {avg_score:.2f}, "
                  f"Avg Reward (last 1000): {avg_reward:.2f}, Epsilon: {epsilon:.3f},\n"
                  f"Exploration ratio last run: {explore_count / total:.3f} (explore:{explore_count} exploit:{exploit_count}), top score : {top_score}"
                  f"BestCurrent10k: {interval_best_score}\n")

        # Save periodically
        if (episode + 1) % 10000 == 0:
            np.save(save_path, Q_table)
            print(f"Q-table saved at episode {episode + 1}")

            # Play a single game with visuals enabled
            if visual == "on":
                print(f"Visualizing AI playing at episode {episode + 1}")
                play_single_game(save_path, Q_table)
                # if interval_best_seed is not None:
                #     print(f"Replaying best run of last 10k (score={interval_best_score})")
                #     replay_best_run(interval_best_seed, interval_best_actions, Q_table)

    # Save final Q-table
    np.save(save_path, Q_table)
    print(f"Training completed. Q-table saved to {save_path}")

    import os
    os.makedirs("learning_state", exist_ok=True)
    np.save("learning_state/episode_lengths.npy", np.array(episode_lengths, dtype=int))
    print("Saved per-episode lengths to learning_state/episode_lengths.npy")
    return scores

# Save score array into a file for graphs.py


# IA_Snake.py — add above play_single_game

def _snapshot(game, total_reward):
    return {
        "snake": list(game.snake),
        "direction": game.direction,
        "green_apples": list(game.green_apples),
        "red_apples": list(game.red_apples),
        "alive": game.alive,
        "direction_changed": game.direction_changed,
        "ate_green": game.ate_green,
        "ate_red": game.ate_red,
        "previous_distance": game.previous_distance,
        "total_reward": total_reward,
    }

def _restore(game, snap):
    game.snake = list(snap["snake"])
    game.direction = snap["direction"]
    game.green_apples = list(snap["green_apples"])
    game.red_apples = list(snap["red_apples"])
    game.alive = snap["alive"]
    game.direction_changed = snap["direction_changed"]
    game.ate_green = snap["ate_green"]
    game.ate_red = snap["ate_red"]
    game.previous_distance = snap["previous_distance"]
    return snap["total_reward"]


def play_single_game(save_path, Q_table):
    """Visualize with step mode:
       - Press S to toggle step mode
       - Right arrow: next step
       - Left arrow: previous step
    """
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Snake AI (S=step, ←/→ navigate)")
    clock = pygame.time.Clock()

    game = SnakeGame()
    game.reset()

    # History
    total_reward = 0.0
    history = [_snapshot(game, total_reward)]  # history[0] = initial state
    history_actions = []   # action taken to reach history[i] (i>=1): history_actions[i-1]
    history_qrows = []     # Q row used for that action
    step_idx = 0           # index into history (0..len(history)-1)
    step_mode = False

    def redraw(idx, with_highlight=True):
        nonlocal total_reward
        # Restore state at idx and draw; print what snake sees
        total_reward = _restore(game, history[idx])
        print_vision(game, idx)
        # Pick highlight from history if available
        if with_highlight and idx > 0 and idx-1 < len(history_actions):
            last_action = history_actions[idx-1]
            qrow = history_qrows[idx-1]
        else:
            last_action = None
            qrow = None
        elapsed = idx * 0.1
        draw_game(screen, game, elapsed, total_reward,
                  last_action=last_action, action_values=qrow)

    def forward_compute_one():
        """Compute one AI step from current end of history, append snapshot, draw."""
        nonlocal total_reward, step_idx
        if not game.alive:
            return
        state = encode_state(game)
        qrow = Q_table[state]
        action = safe_argmax(qrow, game)
        direction = action_to_direction(action)
        game.change_direction(direction)
        game.step()
        r = get_reward(game)
        total_reward += r

        # Save
        history_actions.append(action)
        history_qrows.append(qrow.copy())
        history.append(_snapshot(game, total_reward))
        step_idx = len(history) - 1

        # Draw and print
        print_vision(game, step_idx)
        elapsed = step_idx * 0.1
        draw_game(screen, game, elapsed, total_reward,
                  last_action=action, action_values=qrow)

    # Initial draw
    redraw(step_idx, with_highlight=False)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                # toggle step mode with 'S'
                if event.key == pygame.K_s:
                    step_mode = not step_mode

                elif step_mode and event.key == pygame.K_RIGHT:
                    # Forward 1 step
                    if step_idx == len(history) - 1:
                        forward_compute_one()
                    else:
                        step_idx += 1
                        redraw(step_idx, with_highlight=True)

                elif step_mode and event.key == pygame.K_LEFT:
                    # Back 1 step
                    if step_idx > 0:
                        step_idx -= 1
                        redraw(step_idx, with_highlight=True)

        if not step_mode:
            if game.alive:
                forward_compute_one()
            else:
                redraw(step_idx, with_highlight=True)
            clock.tick(15)
        else:
            clock.tick(30)

    pygame.quit()

# def replay_best_run(seed, actions, Q_table):
#     pygame.init()
#     screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
#     pygame.display.set_caption("Best Run (last 10k eps)")
#     clock = pygame.time.Clock()

#     # Recreate exact randomness for apple placement
#     random.seed(seed)
#     game = SnakeGame()
#     game.reset()

#     total_reward = 0
#     steps = 0

#     for action in actions:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 pygame.quit()
#                 return
#         # Apply recorded action (already 0..3)
#         direction = action_to_direction(action)
#         game.change_direction(direction)
#         game.step()

#         # Optional: recompute reward for display (using same get_reward logic)
#         # (Safe fallback if apples changed exactly same due to seed)
#         # We don't strictly need total_reward; keep simple:
#         draw_game(screen, game, steps * 0.1, total_reward)
#         clock.tick(15)
#         steps += 1
#         if not game.alive:
#             break

#     # Hold a short moment
#     pygame.time.delay(500)
#     pygame.quit()

# --- Add below imports / globals ---
def print_vision(game, step_idx=None):
    """
    Prints a BOARD_SIZE x BOARD_SIZE grid:
      '.' = not in any of the 4 vision rays (hidden)
      '0' = visible empty cell
      'H' = snake head (always visible)
      'S' = visible snake body segment
      'G' = visible green apple
      'R' = visible red apple
    Vision = straight rays UP / DOWN / LEFT / RIGHT from head until wall.
    """
    size = game.board_size
    head_x, head_y = game.snake[0]
    body = set(game.snake[1:])
    greens = set(game.green_apples)
    reds = set(game.red_apples)

    # Start with all hidden
    grid = [['.' for _ in range(size)] for _ in range(size)]

    def reveal(x, y):
        if (x, y) == (head_x, head_y):
            grid[y][x] = 'H'
        elif (x, y) in body:
            grid[y][x] = 'S'
        elif (x, y) in greens:
            grid[y][x] = 'G'
        elif (x, y) in reds:
            grid[y][x] = 'R'
        else:
            grid[y][x] = '0'

    # Head always visible
    reveal(head_x, head_y)

    # Rays: up, down, left, right
    for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
        x, y = head_x, head_y
        while True:
            x += dx
            y += dy
            if x < 0 or x >= size or y < 0 or y >= size:
                break
            reveal(x, y)

    prefix = f"[STEP {step_idx}] " if step_idx is not None else ""
    print(prefix)
    for row in grid:
        print(''.join(row))
    print()  # extra newline



def test_agent(save_path, nb_games=10):
    """Test the trained agent."""
    global Q_table

    try:
        Q_table = np.load(save_path)
        print(f"Loaded Q-table from {save_path}")
    except FileNotFoundError:
        print(f"No Q-table found at {save_path}")
        return

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Snake AI Test")
    clock = pygame.time.Clock()

    scores = []

    for game_num in range(nb_games):
        game = SnakeGame()
        game.reset()
        initial_length = len(game.snake)
        steps = 0

        while game.alive and steps < 1000:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            state = encode_state(game)
            action = safe_argmax(Q_table[state], game)

            direction = action_to_direction(action)
            game.change_direction(direction)
            game.step()
            steps += 1

            elapsed_time = steps * 0.1
            draw_game(screen, game, elapsed_time)
            clock.tick(10)  # Slower for observation

        score = len(game.snake) - initial_length
        scores.append(score)
        print(f"Game {game_num + 1}: Score = {score}, Steps = {steps}")

    print(f"Average score over {nb_games} games: {np.mean(scores):.2f}")
    pygame.quit()
