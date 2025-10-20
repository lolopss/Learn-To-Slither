import argparse
from lobby import run_lobby


def parse_args():
    parser = argparse.ArgumentParser(description="Learn2Slither snake AI")

    # Save file path (legacy)
    parser.add_argument(
        "-save",
        type=str,
        default="./learning_state/q_table.npy",
        help="Path to save the learning state (legacy;\
              overridden by -qtable if provided)"
    )

    # Q-table path (preferred)
    parser.add_argument(
        "-qtable",
        type=str,
        default="./learning_state/q_table.npy",
        help="Path to load/save the Q-table (e.g., \
            learning_state/q_table_1_episode.npy)"
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

    parser.add_argument(
        "-speed",
        type=int,
        default=15,
        help="Visualizer speed (frames per second)"
    )
    return parser.parse_args()

# def game_loop():


def main():
    args = parse_args()

    qtable_path = args.qtable if args.qtable else args.save

    print(f"Sessions : {args.sessions}")
    print(f"Q-table  : {qtable_path}")
    print(f"Visual   : {args.visual}")

    # IA_Snake.train_agent(qtable_path, args.episode, args.visual)
    run_lobby(qtable_path, args.episode, args.visual, args.speed)
    print(f"Save learning state in {qtable_path}")


if __name__ == "__main__":
    main()
# --------------------------------MAIN---------------------------------#
