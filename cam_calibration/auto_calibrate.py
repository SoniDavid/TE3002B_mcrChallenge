import cv2
import numpy as np
import yaml
import argparse
import time

def calibrate(objpoints, imgpoints, gray_shape):
    """
    Performs camera calibration using the collected object and image points.
    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_shape[::-1], None, None)
    return ret, mtx, dist

def main():
    parser = argparse.ArgumentParser(description='Auto Camera Calibration utility.')
    parser.add_argument('--rows', type=int, default=6, help='Number of inner corners rows (default: 6)')
    parser.add_argument('--cols', type=int, default=9, help='Number of inner corners columns (default: 9)')
    parser.add_argument('--square_size', type=float, default=3.0, help='Size of a square in real-world units (default: 3.0)')
    parser.add_argument('--output', type=str, default='camera_calibration.yaml', help='Output YAML file')
    parser.add_argument('--target_frames', type=int, default=30, help='Number of good frames to capture automatically')
    parser.add_argument('--delay', type=float, default=1.0, help='Minimum delay (seconds) between captures')
    parser.add_argument('--blur_threshold', type=float, default=50.0, help='Laplacian variance threshold for blur detection')
    
    args = parser.parse_args()

    # Termination criteria for corner sub-pixel accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points
    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    objp *= args.square_size

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    gray_shape = None

    # Open the camera
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("Error: Cannot open /dev/video2.")
        return

    # Force 640x480 at 30 FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print(f"\n--- Auto Capture Started ---")
    print(f"Aiming for {args.target_frames} frames.")
    print("Move the checkerboard slowly. The script will auto-capture when the image is clear and corners are detected.")
    print("Press 'c' to calculate early, 'q' to quit without saving.")
    
    last_capture_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to receive frame from camera. Exiting.")
            break
        
        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_shape = gray.shape

        # 1. Check blur before doing heavy corner detection
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blurry = lap_var < args.blur_threshold

        ret_corners = False
        corners = None

        # 2. Only look for corners if the image isn't very blurry
        if not blurry:
            ret_corners, corners = cv2.findChessboardCorners(gray, (args.cols, args.rows), None)

        current_time = time.time()
        time_since_last = current_time - last_capture_time

        if ret_corners:
            cv2.drawChessboardCorners(display_frame, (args.cols, args.rows), corners, ret_corners)
            
            # 3. If enough time has passed, capture it
            if time_since_last > args.delay:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                last_capture_time = current_time
                
                print(f"Auto-captured frame {len(imgpoints)}/{args.target_frames} (Sharpness: {lap_var:.1f})")
                
                # Flash the screen green briefly to give feedback
                display_frame[:] = (0, 255, 0)
                
                if len(imgpoints) >= args.target_frames:
                    print(f"\nReached target of {args.target_frames} frames. Starting calibration...")
                    break
            else:
                # Waiting for delay, show ready
                cv2.putText(display_frame, "Hold still...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        else:
            if blurry:
                cv2.putText(display_frame, f"Too Blurry ({lap_var:.1f})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(display_frame, "No Corners Detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.putText(display_frame, f"Captured: {len(imgpoints)}/{args.target_frames}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Auto Camera Calibration', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            if len(imgpoints) >= 5:
                break
            else:
                print(f"Need more frames. Currently have {len(imgpoints)}. Recommended: >= 15")
        elif key == ord('q'):
            print("Exiting without calibration.")
            cap.release()
            cv2.destroyAllWindows()
            return

    # Cleanup windows before running heavy math
    cap.release()
    cv2.destroyAllWindows()

    if len(imgpoints) > 0:
        print(f"\nCalibrating with {len(imgpoints)} frames. This may take a minute...")
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
            
            print(f"\n--- Calibration successful! ---")
            print(f"==> RMS error: {rms:.4f}")
            if rms < 1.0:
                print("    (Great! This is an excellent calibration)")
            elif rms < 2.0:
                print("    (Okay, but could be better. Ensure board is perfectly flat)")
            else:
                print("    (Poor. Please re-run and ensure the board is glued/taped to a rigid flat surface)")
                
            print(f"Results saved to {args.output}")
        except Exception as e:
            print(f"Calibration failed: {e}")
    else:
        print("No valid calibration frames were collected. Calibration aborted.")

if __name__ == '__main__':
    main()
