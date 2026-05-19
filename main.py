import cv2
from ultralytics import YOLO

# 1) Load YOLO model
model = YOLO("yolov8n.pt")

# 2) Open video
cap = cv2.VideoCapture("video.mp4")

# 3) Nice window size (not zoomed)
cv2.namedWindow("Object Detection + Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Object Detection + Tracking", 1100, 650)

# COCO class IDs (common): car=2, bus=5, truck=7
VEHICLE_CLASSES = [2, 5, 7]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame (keeps it consistent and prevents huge window)
    frame = cv2.resize(frame, (960, 540))

    # 4) Track with filters (less boxes, cleaner)
    # conf + iou + classes reduce clutter dramatically
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.60,
        iou=0.50,
        classes=VEHICLE_CLASSES,
        max_det=50
    )

    # Draw on a copy of the frame
    out = frame.copy()

    boxes = results[0].boxes
    names = results[0].names

    if boxes is not None and boxes.xyxy is not None and boxes.cls is not None:
        xyxy = boxes.xyxy.int().cpu().tolist()
        clss = boxes.cls.int().cpu().tolist()

        # IDs exist only in tracking mode
        ids = None
        if boxes.id is not None:
            ids = boxes.id.int().cpu().tolist()

        for i, (x1, y1, x2, y2) in enumerate(xyxy):
            cls = clss[i]
            label_name = names[cls]

            # Box style (thin)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Small label
            if ids is not None:
                label = f"ID {ids[i]} | {label_name}"
            else:
                label = f"{label_name}"

            cv2.putText(
                out,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    cv2.imshow("Object Detection + Tracking", out)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
