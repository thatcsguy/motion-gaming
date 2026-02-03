"""Main entry point for Motion Gaming."""

import cv2

from motion_gaming.gesture_recognizer import count_fingers, recognize_pointing
from motion_gaming.hand_detector import HandDetector
from motion_gaming.input_controller import InputController
from motion_gaming.preview_window import PreviewWindow


def main() -> None:
    """Run the main application loop."""
    # Initialize components
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    detector = HandDetector()
    controller = InputController()
    preview = PreviewWindow()

    is_tracking = True
    print("Motion Gaming started. Press F12 to toggle tracking, ESC to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Mirror the frame for more intuitive interaction
            frame = cv2.flip(frame, 1)

            direction = None
            finger_count = 0

            if is_tracking:
                # Detect hands
                hands = detector.detect(frame)

                # Process each hand
                for hand in hands:
                    if hand.handedness == "Right":
                        # Right hand controls movement (pointing direction)
                        direction = recognize_pointing(hand.landmarks)
                    elif hand.handedness == "Left":
                        # Left hand controls abilities (finger count)
                        finger_count = count_fingers(hand.landmarks)

                # Update input controller
                controller.set_movement(direction)
                controller.set_ability(finger_count)
            else:
                hands = []
                controller.release_all()

            # Update preview
            preview.update(frame, hands, direction, finger_count, is_tracking)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == 123 or key == ord("p"):  # F12 or 'p' for toggle
                is_tracking = not is_tracking
                if not is_tracking:
                    controller.release_all()
                print(f"Tracking: {'ON' if is_tracking else 'OFF'}")

            # Check if window was closed
            if not preview.is_open():
                break

    finally:
        # Cleanup
        controller.release_all()
        detector.close()
        preview.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Motion Gaming stopped.")


if __name__ == "__main__":
    main()
