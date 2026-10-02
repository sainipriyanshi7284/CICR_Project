import cv2
from ultralytics import YOLO

# Load YOLO model
m = YOLO("yolo11n.pt")

# Open Webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("AI Object Tracking & Counting Started!")

window_name = "AI Object Tracking & Counting"

while True:

    # Read frame
    ret, frame = cap.read()

    if not ret:
        print("Cannot receive frame")
        break

    # Detect and Track
    results = m.track(
        frame,
        persist=True,
        verbose=False
    )

    output = frame.copy()

    counts = {}

    if results[0].boxes is not None:

        for box in results[0].boxes:

            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Class information
            cls = int(box.cls[0])
            name = m.names[cls]

            # Count objects
            counts[name] = counts.get(name, 0) + 1

            # Tracking ID
            if box.id is not None:
                track_id = int(box.id[0])
            else:
                track_id = -1

            # Draw rectangle
            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Label
            label = f"{name} | ID:{track_id}"

            # Label Background
            cv2.rectangle(
                output,
                (x1, max(0, y1 - 35)),
                (x1 + 180, y1),
                (0, 0, 0),
                -1
            )

            # Draw Label
            cv2.putText(
                output,
                label,
                (x1 + 5, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # Display Count Information
    y = 30

    for obj_name, count in counts.items():

        cv2.putText(
            output,
            f"{obj_name}: {count}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        y += 35

    # Total count
    total_count = sum(counts.values())

    cv2.putText(
        output,
        f"Total Count: {total_count}",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # Show frame
    cv2.imshow(window_name, output)

    # Keyboard Input
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("Program terminated by user.")
        break
    elif key == ord('Q'):
       print("Program terminated by user.")
       break
       
           # Window Close Detection
    try:
        if cv2.getWindowProperty(
            window_name,
            cv2.WND_PROP_VISIBLE
        ) < 1:
            print("Window closed.")
            break
    except:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

print("Program ended successfully.")