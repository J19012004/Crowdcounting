import cv2
import datetime

# Input video file
video_path = "C:/Users/Administrator/Desktop/bala/forest.mp4"

# Initialize video capture
cap = cv2.VideoCapture(video_path)
background = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if background is None:
        background = gray
        continue

    diff_frame = cv2.absdiff(background, gray)
    threshold_frame = cv2.threshold(diff_frame, 25, 255, cv2.THRESH_BINARY)[1]

    threshold_frame = cv2.dilate(threshold_frame, None, iterations=2)
    contours, _ = cv2.findContours(threshold_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Save detected image
        filename = f"wildlife_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)

    cv2.imshow("Wildlife Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()