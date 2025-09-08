import argparse
import pygame
import time
import random


# --------------------------------SNAKE_GAME---------------------------------#



# --------------------------------MAIN---------------------------------#
def parse_args():
    parser = argparse.ArgumentParser(description="Learn2Slither snake AI")

    # Number of sessions (required)
    parser.add_argument(
        "-sessions",
        type=int,
        default=1,
        help="Number of sessions to run"
    )

    # Save file path (required)
    parser.add_argument(
        "-save",
        type=str,
        default="./learning_state",
        help="Path to save the learning state"
    )

    # Visual mode (optional, default on)
    parser.add_argument(
        "-visual",
        choices=["on", "off"],
        default="on",
        help="Enable or disable visualization (default: on)"
    )

    return parser.parse_args()

# def game_loop():
    

def main():
    args = parse_args()

    print(f"Sessions : {args.sessions}")
    print(f"Save path: {args.save}")
    print(f"Visual   : {args.visual}")

    # --- placeholder for actual game logic ---
    max_length = 4
    max_duration = 17

    print(f"Game over, max length = {max_length}, "
          f"max duration = {max_duration}")
    print(f"Save learning state in {args.save}")


if __name__ == "__main__":
    main()
# --------------------------------MAIN---------------------------------#
