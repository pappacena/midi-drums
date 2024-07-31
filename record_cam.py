import cv2

width = 320
height = 240
fps = 130
seconds = 30

cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, fps)

size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('recording.avi', fourcc, fps, size)

for i in range(fps * seconds):
    _, frame = cap.read()
    #cv2.imshow('Recording...', frame)
    out.write(frame)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

cap.release()
out.release()
cv2.destroyAllWindows()
