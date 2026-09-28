import cv2
from ultralytics import YOLO
# Load the AI model
m = YOLO("yolo11n.pt")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("AI Object Counting Started!")    
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Cannot receive frame")
        break
     # Detect objects
    results = m(frame, verbose=False)
    # Draw detection boxes in one frame and store it in output variable
    output = results[0].plot()
    counts = {}
   # count the number of objects detected in the frame
    for box in results[0].boxes:    

        cls = int(box.cls[0])
        name = m.names[cls]
        if name in counts:
            counts[name] += 1
        else:
            counts[name] = 1
    y = 30
    # Display counts on the video
    for name, count in counts.items():
        text = f"{name}: {count}"
        cv2.putText(
            output,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        y += 30
    # Display total count of objects detected in the frame
    total_count = sum(counts.values())
    cv2.putText(
        output,
        f"Total Count: {total_count}",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        2,
    )
    cv2.imshow("AI Object Counting", output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()