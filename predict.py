"""
predict.py

SalesCode prediction script.

Usage:
python predict.py image.jpg

Output:
0.00 -> Real
1.00 -> Screen
"""

import sys
from src.inference import predict

def main():

    if len(sys.argv) != 2:
        print("Usage: python predict.py image.jpg")
        sys.exit(1)

    _, probability, _ = predict(sys.argv[1])

    print(f"{probability:.4f}")

if __name__ == "__main__":
    main()