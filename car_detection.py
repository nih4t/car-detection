import cv2

car_cascade = cv2.CascadeClassifier('cars.xml')
if car_cascade.empty():
    raise Exception('Error loading Haar cascade file')

video = 'cars.mp4'
cap = cv2.VideoCapture(video)

if not cap.isOpened():
    raise Exception("Error opening video file.")


car_detected = False
detection_time = 0

while True:
    ret, frames = cap.read()

    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

    for (x, y, w, h) in cars:
        cv2.rectangle(frames, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.imshow('video', frames)
        crop_img = frames[y:y + h, x:x + w]

    cv2.imshow('video', frames)
    if cv2.waitKey(33) == 27:
        break

cap.release()
cv2.destroyAllWindows()
