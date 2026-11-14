"""
Training Script for Multi-Digit Handwritten Number Recognition AI
Trains a high-performance CRNN (CNN + BiLSTM + CTC) on multi-digit sequences
derived from the official MNIST benchmark, evaluates exact sequence accuracy,
character accuracy, CER, and persists model & metrics.
"""

import sys
import os
import argparse
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.model import train_and_evaluate_multidigit, MODEL_PATH, METRICS_PATH


def main():
    parser = argparse.ArgumentParser(description="Train CRNN for Multi-Digit Handwritten Number Recognition")
    parser.add_argument("--epochs", type=int, default=6, help="Number of training epochs (default: 6)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--num-train", type=int, default=18000, help="Number of training sequences (default: 18000)")
    parser.add_argument("--num-test", type=int, default=2500, help="Number of evaluation sequences (default: 2500)")
    parser.add_argument("--force", action="store_true", help="Force retrain even if model exists")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    print("=" * 65)
    print(" MULTI-DIGIT HANDWRITTEN NUMBER RECOGNITION - TRAINING ")
    print("=" * 65)
    print(f"Target Model:      {MODEL_PATH}")
    print(f"Metrics Path:      {METRICS_PATH}")
    print(f"Epochs:            {args.epochs}")
    print(f"Batch Size:        {args.batch_size}")
    print(f"Train Sequences:   {args.num_train}")
    print(f"Test Sequences:    {args.num_test}")
    print(f"Force Retrain:     {args.force}")
    print("-" * 65)

    start_time = time.time()
    metrics = train_and_evaluate_multidigit(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_train=args.num_train,
        num_test=args.num_test,
        force_retrain=args.force,
        seed=args.seed
    )
    elapsed = time.time() - start_time

    print("\n" + "=" * 65)
    print(" TRAINING & EVALUATION COMPLETE")
    print("=" * 65)
    print(f"Exact Sequence Acc:  {metrics['exact_sequence_accuracy'] * 100:.2f}%")
    print(f"Character-Level Acc: {metrics['character_level_accuracy'] * 100:.2f}%")
    print(f"Character Error Rate:{metrics['character_error_rate_cer'] * 100:.2f}%")
    print(f"Total Time:          {elapsed:.1f} seconds")
    print(f"Model saved to:      {MODEL_PATH}")
    print(f"Metrics saved to:    {METRICS_PATH}")
    print("=" * 65)


if __name__ == "__main__":
    main()
