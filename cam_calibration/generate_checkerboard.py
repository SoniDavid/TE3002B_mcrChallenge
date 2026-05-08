import cv2
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description='Generate a checkerboard pattern for camera calibration.')
    parser.add_argument('--width', type=int, default=10, help='Number of squares horizontally (default: 10)')
    parser.add_argument('--height', type=int, default=7, help='Number of squares vertically (default: 7)')
    parser.add_argument('--square_size', type=int, default=100, help='Size of each square in pixels (default: 100)')
    parser.add_argument('--output', type=str, default='checkerboard.png', help='Output filename (default: checkerboard.png)')
    
    args = parser.parse_args()

    # Create the board
    # width x height squares
    board = np.zeros((args.height * args.square_size, args.width * args.square_size), dtype=np.uint8)

    for i in range(args.height):
        for j in range(args.width):
            if (i + j) % 2 == 0:
                board[i*args.square_size:(i+1)*args.square_size, j*args.square_size:(j+1)*args.square_size] = 255

    cv2.imwrite(args.output, board)
    print(f"Generated a {args.width}x{args.height} checkerboard.")
    print(f"Inner corners will be {args.width-1}x{args.height-1}.")
    print(f"Saved to {args.output}")

if __name__ == '__main__':
    main()
