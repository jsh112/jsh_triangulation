import cv2
import numpy as np
from pathlib import Path

from config import *
import stereo_utils as su

def main():
    if not Path(NPZ_PATH).exists():
        raise FileNotFoundError(f"File not found: {NPZ_PATH}")
    
    # Ready
    map1x, map1y, map2x, map2y, P1, P2, size, B, M = su.load_stereo(NPZ_PATH)
    W, H = size
    print(f"[Info] image_size={(W,H)}, baseline~{B:.2f} mm")

        # 레이저 원점 O = LEFT 카메라 원점 L + (왼 1.85cm, 위 8cm, 뒤 3.3cm)
    L = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # 왼쪽 카메라가 원점
    dx = -LASER_OFFSET_CM_LEFT * 10.0
    dy = (-1.0 if Y_UP_IS_NEGATIVE else 1.0) * LASER_OFFSET_CM_UP * 10.0
    dz = LASER_OFFSET_CM_FWD * 10.0
    O  = L + np.array([dx, dy, dz], dtype=np.float64)
    print(f"[Laser] Origin O (mm, LEFT-based) = {O}")

    # 색상 필터 선택
    if SELECTED_COLOR is not None:
        sc = SELECTED_COLOR.strip().lower()
        selected_class_name = ALL_COLORS.get(sc)
        if selected_class_name is None:
            print(f"[Filter] SELECTED_COLOR='{SELECTED_COLOR}' 인식 실패. 콘솔에서 선택합니다.")
            selected_class_name = su.ask_color_and_map_to_class(ALL_COLORS)
        else:
            print(f"[Filter] 선택 클래스(상수): {selected_class_name}")
    else:
        selected_class_name = su.ask_color_and_map_to_class(ALL_COLORS)

    # 카메라 & 모델
    capL_idx, capR_idx = CAM1_INDEX, CAM2_INDEX
    if SWAP_INPUT:
        capL_idx, capR_idx = capR_idx, capL_idx  # 입력을 스왑하여 보정 좌/우와 일치
    cap1, cap2 = su.open_cams(capL_idx, capR_idx, size)

    # 최초 두 프레임 버리고 INIT_DET_FRAMES 만큼 저장
    su.save_rectified_frames(cap1, cap2, map1x, map1y, map2x, map2y, size,
                      INIT_DET_FRAMES, Path("rectified_frames_L"), Path("rectified_frames_R"))
    


if __name__ == "__main__":
    main()