import numpy as np
import random
import pygame
from game import SnakeGame, draw_game, WINDOW_WIDTH, WINDOW_HEIGHT, BOARD_SIZE

# Improved state space - much smaller and more focused
state_space_size = 16 * 9 * 4  # 576 states
action_space_size = 4     # Up, Down, Left, Right

Q_table = np.zeros((state_space_size, action_space_size))

# Improved hyperparameters
alpha = 0.1         # Slightly higher learning rate
gamma = 0.9         # Higher discount factor - care more about future
epsilon = 0.50        # Exploration rate
epsilon_decay = 0.995  # Slower decay
min_epsilon = 0.01   # Lower minimum exploration
explore_count = 0
exploit_count = 0

# To pick best game
interval_best_score = -1
interval_best_actions = []
interval_best_seed = None
# -------------------------------- STATE ENCODING ----------------------------- #


def encode_state(game):
    # Danger bits
    head_x, head_y = game.snake[0]
    dx, dy = game.direction
    dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    current_dir_idx = dirs.index(game.direction)

    def is_danger(nx, ny):
        return (nx < 0 or nx >= game.board_size or
                ny < 0 or ny >= game.board_size or
                (nx, ny) in game.snake)

    danger_ahead = int(is_danger(head_x + dx, head_y + dy))
    # left/right relative rotations
    max_length = 0
    left_dx, left_dy = -dy, dx
    right_dx, right_dy = dy, -dx
    danger_left = int(is_danger(head_x + left_dx, head_y + left_dy))
    danger_right = int(is_danger(head_x + right_dx, head_y + right_dy))
    danger_behind = int(is_danger(head_x - dx, head_y - dy))

    danger_bits = (danger_ahead << 3) | (danger_left << 2) | (danger_right << 1) | danger_behind  # 0..15

    # Apple direction (coarse 3x3)
    if game.green_apples:
        # nearest
        ax, ay = min(game.green_apples,
                     key=lambda a: abs(a[0]-head_x)+abs(a[1]-head_y))
    else:
        ax, ay = head_x, head_y
    apple_dir_x = 0 if ax < head_x else (2 if ax > head_x else 1)  # 0 left,1 same,2 right
    apple_dir_y = 0 if ay < head_y else (2 if ay > head_y else 1)  # 0 up,1 same,2 down
    apple_index = apple_dir_y * 3 + apple_dir_x  # 0..8

    # Combine: (((danger_bits)*9)+apple_index)*4 + current_dir
    state = ((danger_bits * 9) + apple_index) * 4 + current_dir_idx  # Range 0..575
    return state


# ----------------------- REWARD FUNCTION ----------------- #

def get_reward(game):
    if not game.alive:
        return -100  # Game over
    elif game.ate_green == 1:
        return 10  # Green apple
    elif game.ate_red == 1:
        return -10  # Red apple
    else:
        # Reward for moving closer to the green apple
        head_x, head_y = game.snake[0]
        apple_x, apple_y = game.green_apples[0]
        current_distance = abs(head_x - apple_x) + abs(head_y - apple_y)

        if current_distance < game.previous_distance:
            reward = 1  # Reward for getting closer
        elif current_distance > game.previous_distance:
            reward = -1  # Penalty for getting farther
        else:
            reward = -1  # Small penalty for staying the same

        game.previous_distance = current_distance
        return reward

# -------------------------------- TRAINING LOOP ------------------------------- #


def choose_action(state, Q_table, epsilon):
    global explore_count, exploit_count
    # Current direction to block 180° turn (game enforces too)
    # But blocking here avoids wasted random attempts
    # Map index back to direction for opposite detection
    action_dir_map = [(0,-1),(0,1),(-1,0),(1,0)]
    # We'll infer current direction from best action row if needed
    # (Better: pass game.direction into this function)
    row = Q_table[state]
    # Exploration
    if random.random() < epsilon:
        explore_count += 1
        # Sample valid actions (exclude exact opposite of current)
        # Pass current direction separately instead of decoding
        # Adjust signature if you refactor:
        return random.randint(0,3)
    else:
        exploit_count += 1
        return int(np.argmax(row))


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
            action = choose_action(state, Q_table, epsilon)
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
        epsilon = max(min_epsilon, epsilon * 0.9995)

        top_score = max(top_score, final_score)
        # Print progress
        if (episode + 1) % 1000 == 0:
            avg_score = np.mean(recent_scores) if recent_scores else 0
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0
            total = explore_count + exploit_count
            print(f"Episode {episode + 1}: Avg length (last 1000): {avg_score:.2f}, "
                  f"Avg Reward (last 1000): {avg_reward:.2f}, Epsilon: {epsilon:.3f},\n"
                  f"Exploration ratio last run: {explore_count / total:.3f} (explore:{explore_count} exploit:{exploit_count}), top score : {top_score}"
                  f"BestCurrent10k: {interval_best_score}")

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

    return scores


def play_single_game(save_path, Q_table):
    """Play a single game with visuals enabled."""

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Snake AI Playing")
    clock = pygame.time.Clock()

    game = SnakeGame()
    game.reset()
    steps = 0
    total_reward = 0
    while game.alive and steps < 1000:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Get the current state and choose the best action
        state = encode_state(game)
        action = np.argmax(Q_table[state])  # Always choose the best action
        direction = action_to_direction(action)
        game.change_direction(direction)
        game.step()

        # Draw the game
        elapsed_time = steps * 0.1  # Approximate elapsed time
        draw_game(screen, game, elapsed_time, total_reward)
        clock.tick(10)  # Slower frame rate for observation

        steps += 1
    pygame.quit()  # Close the window after the game ends

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
            action = np.argmax(Q_table[state])  # Choose best action

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
