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
min_epsilon = 0.1   # Lower minimum exploration


# -------------------------------- STATE ENCODING ----------------------------- #


def encode_state(game):
    """
    Encode game state into a single integer for Q-table indexing.

    State features:
    - Danger ahead (1 bit): Is there danger (wall/body) in front?
    - Danger left (1 bit): Is there danger to the left?
    - Danger right (1 bit): Is there danger to the right?
    - Danger behind (1 bit): Is there danger behind?
    - Apple direction X (2 bits): 0=left, 1=center, 2=right
    - Apple direction Y (2 bits): 0=up, 1=center, 2=down
    - Current direction (2 bits): 0=up, 1=down, 2=left, 3=right
    """
    head_x, head_y = game.snake[0]
    dx, dy = game.direction

    # Get closest green apple
    if not game.green_apples:
        apple_x, apple_y = head_x, head_y  # Fallback
    else:
        apple_x, apple_y = min(game.green_apples,
                               key=lambda apple: abs(apple[0] - head_x)
                               + abs(apple[1] - head_y))

    # Check for dangers
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right
    current_dir_idx = directions.index(game.direction)

    def is_danger(pos_x, pos_y):
        return (pos_x < 0 or pos_x >= game.board_size or
                pos_y < 0 or pos_y >= game.board_size or
                (pos_x, pos_y) in game.snake)

    danger_ahead = int(is_danger(head_x + dx, head_y + dy))
    danger_left = int(is_danger(head_x - dy, head_y + dx))  # Rotate left
    danger_right = int(is_danger(head_x + dy, head_y - dx))  # Rotate right
    danger_behind = int(is_danger(head_x - dx, head_y - dy))

    # Apple direction relative to head
    apple_dir_x = 1 if apple_x == head_x else (0 if apple_x < head_x else 2)
    apple_dir_y = 1 if apple_y == head_y else (0 if apple_y < head_y else 2)

    # Encode state as a single integer
    state = (danger_ahead * (2**6) +
             danger_left * (2**5) +
             danger_right * (2**4) +
             danger_behind * (2**3) +
             apple_dir_x * (2**2) +
             apple_dir_y * (2**1) +
             current_dir_idx)

    return min(state, state_space_size - 1)  # Ensure within bounds


# ----------------------- REWARD FUNCTION ----------------- #

def get_reward(game):
    if not game.alive:
        return -100  # Game over
    elif game.ate_green == 1:
        return 50  # Green apple
    elif game.ate_red == 1:
        return -50  # Red apple
    else:
        # Reward for moving closer to the green apple
        head_x, head_y = game.snake[0]
        apple_x, apple_y = game.green_apples[0]
        current_distance = abs(head_x - apple_x) + abs(head_y - apple_y)

        if current_distance < game.previous_distance:
            reward = 5  # Reward for getting closer
        elif current_distance > game.previous_distance:
            reward = -5  # Penalty for getting farther
        else:
            reward = -1  # Small penalty for staying the same

        game.previous_distance = current_distance
        return reward

# -------------------------------- TRAINING LOOP ------------------------------- #

def train_agent(save_path, nb_episodes, visual):
    global Q_table
    global epsilon

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

    # Training loop
    for episode in range(nb_episodes):
        game = SnakeGame()
        game.reset()
        state = encode_state(game)
        total_reward = 0  # Track total reward for the current episode
        steps = 0
        initial_length = len(game.snake)

        while game.alive and steps < 1000:  # Prevent infinite loops
            # Choose action (exploration vs exploitation)
            action = choose_action(state, Q_table, epsilon)

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
        scores.append(final_score)
        recent_scores.append(final_score)
        recent_rewards.append(total_reward)  # Add total reward for this episode

        if len(recent_scores) > 100:
            recent_scores.pop(0)

        if len(recent_rewards) > 100:
            recent_rewards.pop(0)

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # Print progress
        if (episode + 1) % 1000 == 0:
            avg_score = np.mean(recent_scores) if recent_scores else 0
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0
            print(f"Episode {episode + 1}: Avg length (last 100): {avg_score:.2f}, "
                  f"Avg Reward (last 100): {avg_reward:.2f}, Epsilon: {epsilon:.3f}")

        # Save periodically
        if (episode + 1) % 10000 == 0:
            np.save(save_path, Q_table)
            print(f"Q-table saved at episode {episode + 1}")

            # Play a single game with visuals enabled
            if visual == "on":
                print(f"Visualizing AI playing at episode {episode + 1}")
                play_single_game_with_reopen(save_path, Q_table)

    # Save final Q-table
    np.save(save_path, Q_table)
    print(f"Training completed. Q-table saved to {save_path}")

    return scores


def play_single_game_with_reopen(save_path, Q_table):
    """Play a single game with visuals enabled, ensuring the window closes and reopens."""
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


def choose_action(state, Q_table, epsilon):
    """Improved action selection with better exploration."""
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 3)  # Random action (exploration)
    else:
        return np.argmax(Q_table[state])  # Best action (exploitation)


def action_to_direction(action):
    """Convert action index to direction tuple."""
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right
    return directions[action]


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
            action = np.argmax(Q_table[state])  # Always choose best action

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
