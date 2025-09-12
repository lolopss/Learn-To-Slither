import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys


def rolling(arr, k):
    if k <= 1 or k > len(arr):
        return arr, np.arange(len(arr))
    sm = np.convolve(arr, np.ones(k), 'valid') / k
    return sm, np.arange(k-1, len(arr))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="learning_state/episode_lengths.npy",
                        help="Path to saved per-episode lengths")
    parser.add_argument("--smooth", type=int, default=0,
                        help="Rolling window size (0 = no smoothing)")
    args = parser.parse_args()

    # Resolve path relative to this script for reliability
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, args.path)

    if not os.path.isfile(full_path):
        print(f"[ERROR] File not found: {full_path}")
        sys.exit(1)

    try:
        data = np.load(full_path)
    except Exception as e:
        print(f"[ERROR] Could not load file: {e}")
        sys.exit(1)

    # Debug info
    print(f"[INFO] Loaded: {full_path}")
    print(f"[INFO] Shape: {data.shape}, dtype: {data.dtype}")

    # Normalize shape
    if data.ndim > 1:
        data = data.ravel()
        print(f"[INFO] Flattened to: {data.shape}")

    if data.size == 0:
        print("[WARN] Array is empty. Nothing to plot.")
        sys.exit(0)

    # Remove NaNs if any
    if np.isnan(data).any():
        print("[WARN] NaNs detected. Removing.")
        data = data[~np.isnan(data)]

    if data.size == 0:
        print("[WARN] All values were NaN after filtering.")
        sys.exit(0)

    print(f"[INFO] First 10 values: {data[:10]}")
    print(f"[INFO] Min={data.min()} Max={data.max()} Mean={data.mean():.2f}")

    episodes = np.arange(1, len(data)+1)

    plt.figure(figsize=(12, 6))
    plt.plot(episodes, data, linewidth=0.7, alpha=0.8, label="Episode length",
             color="#4db870")

    if args.smooth > 1 and len(data) >= args.smooth:
        smoothed, sm_eps = rolling(data, args.smooth)
        plt.plot(sm_eps+1, smoothed, color="#d95f02", linewidth=2,
                 label=f"Rolling mean (k={args.smooth})")

    # If all values are identical, add a note
    if data.min() == data.max():
        plt.text(0.5, 0.9, "All values identical",
                 transform=plt.gca().transAxes, ha="center", color="red")

    plt.xlabel("Episode")
    plt.ylabel("Final snake length")
    plt.title("Snake Training Progress (Per Episode)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
