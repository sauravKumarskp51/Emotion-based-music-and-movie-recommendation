import cv2

def test_webcam(index=0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Webcam at index {index} not found")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Webcam Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

test_webcam(0)  # Test the default webcam
test_webcam(1)  # Test the USB webcam, if available
