import cv2
import numpy as np
import yaml
import argparse
import glob
import os

def calibrate(objpoints, imgpoints, gray_shape):
    """
    Performs camera calibration using the collected object and image points.
    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_shape[::-1], None, None)
    return ret, mtx, dist

def main():
    parser = argparse.ArgumentParser(description='Camera calibration utility for TE3002B mcrChallenge.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--online', action='store_true', help='Live capture from /dev/video2. Controls: "s" to capture frame, "c" to calibrate, "q" to quit.')
    group.add_argument('--offline', type=str, help='Path to directory containing calibration images (PNG/JPG).')
    
    parser.add_argument('--rows', type=int, default=6, help='Number of inner corners rows (default: 6)')
    parser.add_argument('--cols', type=int, default=9, help='Number of inner corners columns (default: 9)')
    parser.add_argument('--square_size', type=float, default=1.0, help='Size of a square in real-world units (e.g. mm or meters, default: 1.0)')
    parser.add_argument('--output', type=str, default='camera_calibration.yaml', help='Output YAML file (default: camera_calibration.yaml)')
    
    args = parser.parse_args()

    # Termination criteria for corner sub-pixel accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(cols-1,rows-1,0)
    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    objp *= args.square_size

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    gray_shape = None

    if args.offline:
        if not os.path.isdir(args.offline):
            print(f"Error: {args.offline} is not a directory.")
            return
            
        images = sorted(glob.glob(os.path.join(args.offline, '*.png')) + \
                        glob.glob(os.path.join(args.offline, '*.jpg')) + \
                        glob.glob(os.path.join(args.offline, '*.jpeg')))
        
        if not images:
            print(f"No images found in {args.offline}")
            return

        print(f"Found {len(images)} images. Processing...")
        for fname in images:
            img = cv2.imread(fname)
            if img is None:
                print(f"Could not read {fname}")
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_shape = gray.shape

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (args.cols, args.rows), None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                print(f"Corners detected in {fname}")
            else:
                print(f"Failed to detect corners in {fname}")

    elif args.online:
        # Attempt to open /dev/video2
        cap = cv2.VideoCapture(2)
        if not cap.isOpened():
            print("Error: Cannot open /dev/video2. Check if the device exists and you have permissions.")
            return

        print("Live Capture Started.")
        print("Instructions:")
        print("  - Press 's' to capture current frame for calibration (only if corners are detected)")
        print("  - Press 'c' to run calibration with captured frames and exit")
        print("  - Press 'q' to quit without calibrating")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to receive frame from camera. Exiting.")
                break
            
            display_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_shape = gray.shape

            # Find corners to show user if detection is working
            ret_corners, corners = cv2.findChessboardCorners(gray, (args.cols, args.rows), None)

            if ret_corners:
                cv2.drawChessboardCorners(display_frame, (args.cols, args.rows), corners, ret_corners)
                cv2.putText(display_frame, f"Captured: {len(imgpoints)}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "No Corners Detected", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('Camera Calibration - Online Mode', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                if ret_corners:
                    objpoints.append(objp)
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    imgpoints.append(corners2)
                    print(f"Captured frame {len(imgpoints)}")
                else:
                    print("Corners not detected, frame not captured.")
            elif key == ord('c'):
                if len(imgpoints) >= 5:
                    break
                else:
                    print(f"Need more frames. Currently have {len(imgpoints)}. Recommended: >= 10")
            elif key == ord('q'):
                print("Exiting without calibration.")
                cap.release()
                cv2.destroyAllWindows()
                return

        cap.release()
        cv2.destroyAllWindows()

    if len(imgpoints) > 0:
        print(f"Calibrating with {len(imgpoints)} frames...")
        try:
            rms, mtx, dist = calibrate(objpoints, imgpoints, gray_shape)
            
            data = {
                'rms_error': float(rms),
                'camera_matrix': mtx.tolist(),
                'dist_coeff': dist.tolist(),
                'img_shape': list(gray_shape)
            }

            with open(args.output, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
            
            print(f"Calibration successful. RMS error: {rms:.4f}")
            print(f"Results saved to {args.output}")
        except Exception as e:
            print(f"Calibration failed: {e}")
    else:
        print("No valid calibration frames were collected. Calibration aborted.")

if __name__ == '__main__':
    main()
