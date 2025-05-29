import argparse
import time
import cv2
import numpy as np

# Load video
cap = cv2.VideoCapture("/content/drive/MyDrive/Hasan_ Kalyoncu_University/Route4-11-11-2020.mp4")
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

time.sleep(0.1)

# Background subtractor for dynamic scenes
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

# Shi-Tomasi corner detection for optical flow
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Read first frame
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for speed (optional)
    frame = cv2.resize(frame, (640, 480))

    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)

    # Morphological operations to clean the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    # Find contours in foreground mask
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < 100:
            continue
        x, y, w, h = cv2.boundingRect(c)

        # Color bounding box based on size
        color = (0, 0, 255) if w * h > 2000 else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Optical Flow to track motion
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, prev_pts, None, **lk_params)

    if next_pts is not None and prev_pts is not None:
        for i, (new, old) in enumerate(zip(next_pts, prev_pts)):
            a, b = new.ravel()
            c, d = old.ravel()
            motion_vector = np.sqrt((a - c)**2 + (b - d)**2)
            if motion_vector > 2.0:
                cv2.arrowedLine(frame, (int(c), int(d)), (int(a), int(b)), (255, 0, 0), 1)

    prev_gray = frame_gray.copy()
    prev_pts = next_pts

    # Display
    cv2.imshow("Foreground Mask", fg_mask)
    cv2.imshow("Obstacle Detection with Optical Flow", frame)

    key = cv2.waitKey(30)
    if key == 27:
        break
    elif key == ord('s'):
        cv2.imwrite('frame.png', frame)
        cv2.imwrite('mask.png', fg_mask)

cap.release()
cv2.destroyAllWindows()
