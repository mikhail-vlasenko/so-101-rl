import sys

import cv2

from real.vision.camera import open_camera

OUTPUT = "c922_frame.jpg"

cap = open_camera()
ok, frame = cap.read()
cap.release()

if not ok:
    print("Failed to read frame", file=sys.stderr)
    sys.exit(1)

cv2.imwrite(OUTPUT, frame)
print(f"Saved {OUTPUT} ({frame.shape[1]}x{frame.shape[0]})")
