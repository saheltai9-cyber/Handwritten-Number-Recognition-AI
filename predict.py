"""
Command-Line Prediction Interface for Multi-Digit Handwritten Number Recognition AI
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.inference import predict_number


def main():
    parser = argparse.ArgumentParser(description="Recognize handwritten multi-digit numbers from an image file")
    parser.add_argument("image_path", type=str, help="Path to number image (PNG, JPG, JPEG)")
    parser.add_argument("--threshold", type=float, default=0.35, help="Uncertainty threshold (default: 0.35)")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        sys.exit(1)

    result = predict_number(args.image_path, uncertainty_threshold=args.threshold)

    print("=" * 60)
    print(" MULTI-DIGIT HANDWRITTEN NUMBER RECOGNITION RESULT ")
    print("=" * 60)
    print(f"File:         {args.image_path}")
    print(f"SHA-256:      {result['sha256_hash'][:16]}...")
    print(f"Status:       {'SUCCESS' if result['success'] else ('UNCERTAIN' if result.get('uncertain') else 'FAILED')}")
    print(f"Message:      {result['message']}")

    if result.get("predicted_number") is not None:
        print(f"\nRecognized Number:  {result['predicted_number']}")
        print(f"Model Confidence:   {result['confidence'] * 100:.2f}%")
        
        digits_info = result.get("per_digit_details", [])
        if digits_info:
            print("\nPer-Digit Breakdown:")
            for idx, item in enumerate(digits_info):
                bar = "#" * int(round(item['confidence'] * 20))
                print(f"  Pos {idx + 1} -> Digit {item['digit']}: {item['confidence'] * 100:5.1f}%  [{bar:<20}]")
    print("=" * 60)


if __name__ == "__main__":
    main()
