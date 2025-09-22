# camera_test.py
import cv2

def find_and_show_camera(max_id=10):
    cap = None
    cam_id = None

    # 1~max_id까지 카메라 탐색 (0번은 제외)
    for i in range(1, max_id + 1):
        temp_cap = cv2.VideoCapture(i)
        if temp_cap.isOpened():
            cam_id = i
            cap = temp_cap
            print(f"✅ Found camera at ID {cam_id}")
            break
        temp_cap.release()

    if cap is None:
        print("❌ No external cameras found")
        return

    print(f"🎥 Showing camera {cam_id}. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        cv2.imshow(f"Camera {cam_id}", frame)

        # q 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    find_and_show_camera()