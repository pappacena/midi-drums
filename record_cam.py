import time
import cv2

width = 640
height = 480
fps = 60
seconds = 15

cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, fps)

size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('recording.avi', fourcc, fps, size)

for i in range(5):
    print(f"Recording in {5 - i}...")
    time.sleep(1)

prev_sec = int(time.time())
frames_recorded = 0
for i in range(fps * seconds):
    curr_sec = int(time.time())
    if curr_sec != prev_sec:
        print(f"Recorded {frames_recorded}")
        frames_recorded = 0
    prev_sec = curr_sec

    _, frame = cap.read()
    #cv2.imshow('Recording...', frame)
    out.write(frame)
    frames_recorded += 1
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

cap.release()
out.release()
