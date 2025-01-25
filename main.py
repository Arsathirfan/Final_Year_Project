import cv2

print("Helo")

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    _, frame = camera.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
