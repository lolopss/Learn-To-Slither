import argparse
import time
import random
import game
import IA_Snake
from lobby import run_lobby
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
        default="./learning_state/q_table.npy",
        help="Path to save the learning state"
    )

    # Visual mode (optional, default on)
    parser.add_argument(
        "-visual",
        choices=["on", "off"],
        default="on",
        help="Enable or disable visualization (default: on)"
    )

    parser.add_argument(
        "-training",
        choices=["on", "off"],
        default="on",
        help="Enable or disable IA training mode"
    )

    parser.add_argument(
        "-episode",
        type=int,
        default=1001,
        help="Amount of episodes to train AI"
    )
    return parser.parse_args()

# def game_loop():


def main():
    args = parse_args()

    print(f"Sessions : {args.sessions}")
    print(f"Save path: {args.save}")
    print(f"Visual   : {args.visual}")

    
    # IA_Snake.train_agent(args.save, args.episode, args.visual)
    run_lobby()
    print(f"Save learning state in {args.save}")


if __name__ == "__main__":
    main()
# --------------------------------MAIN---------------------------------#
