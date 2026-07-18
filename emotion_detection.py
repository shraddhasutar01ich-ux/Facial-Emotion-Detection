import cv2
import csv
from datetime import datetime
from fer import FER

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize FER detector
detector = FER(mtcnn=True)

# Create/overwrite CSV file for saving results
csv_file = open("emotion_log.csv", mode="w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "Dominant Emotion", "All Emotions"])

print("Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect emotions
    results = detector.detect_emotions(frame)

    for result in results:
        (x, y, w, h) = result["box"]
        emotions = result["emotions"]

        # Find dominant emotion
        dominant_emotion = max(emotions, key=emotions.get)

        # Draw rectangle and emotion text on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, dominant_emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Save to CSV
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        csv_writer.writerow([timestamp, dominant_emotion, emotions])

    # Show video feed
    cv2.imshow("Emotion Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
csv_file.close()

print("Emotion detection stopped. Data saved to emotion_log.csv")
