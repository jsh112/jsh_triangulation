import numpy as np
import cv2
from pathlib import Path
from config import *

def load_stereo(npz_path):
    S = np.load(npz_path, allow_pickle=True)
    K1, D1 = S["K1"], S["D1"]; K2, D2 = S["K2"], S["D2"]
    R1, R2 = S["R1"], S["R2"]; P1, P2 = S["P1"], S["P2"]
    W, H   = [int(x) for x in S["image_size"]]
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (W, H), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (W, H), cv2.CV_32FC1)
    Tx = -P2[0,3] / P2[0,0]
    B  = float(abs(Tx))
    M  = np.array([0.5*Tx, 0.0, 0.0], dtype=np.float64)  # 중점(정보용)
    return (map1x, map1y, map2x, map2y, P1, P2, (W, H), B, M)

def ask_color_and_map_to_class(all_colors_dict):
    print("🎨 선택 가능한 색상:", ", ".join(all_colors_dict.keys()))
    s = input("✅ 원하는 홀드 색상 입력(엔터=전체): ").strip().lower()
    if not s:
        print("→ 전체 클래스 사용"); return None
    mapped = all_colors_dict.get(s)
    if mapped is None:
        print(f"⚠️ '{s}' 는 유효하지 않은 색상입니다. 전체 클래스 사용")
        return None
    print(f"🎯 선택된 클래스: {mapped}")
    return mapped

def open_cams(idx1, idx2, size):
    W, H = size
    cap1 = cv2.VideoCapture(idx1, cv2.CAP_DSHOW)
    cap2 = cv2.VideoCapture(idx2, cv2.CAP_DSHOW)
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH,  W); cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH,  W); cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    if not cap1.isOpened() or not cap2.isOpened():
        raise SystemExit("Error In open_cams(), 카메라를 열 수 없습니다. 인덱스/연결 확인.")
    return cap1, cap2

def rectify(frame, mx, my, size):
    W, H = size
    if (frame.shape[1], frame.shape[0]) != (W, H):
        frame = cv2.resize(frame, (W, H))
    return cv2.remap(frame, mx, my, cv2.INTER_LINEAR)

def save_rectified_frames(cap1, cap2, map1x, map1y, map2x, map2y, size, n_frames, save_dir_L, save_dir_R):
    save_dir_L.mkdir(parents=True, exist_ok=True)
    save_dir_R.mkdir(parents=True, exist_ok=True)

    # 워밍업
    for _ in range(2):
        cap1.read(); cap2.read()

    for k in range(n_frames):
        ok1, f1 = cap1.read()
        ok2, f2 = cap2.read()
        if not (ok1 and ok2):
            cap1.release(); cap2.release()
            raise SystemExit("초기 프레임 캡처 실패")

        Lr_k = rectify(f1, map1x, map1y, size)
        Rr_k = rectify(f2, map2x, map2y, size)

        cv2.imwrite(str(save_dir_L / f"L_{k:04d}.png"), Lr_k)
        cv2.imwrite(str(save_dir_R / f"R_{k:04d}.png"), Rr_k)
        print(f"[Saved] Frame {k+1}: {save_dir_L / f'L_{k:04d}.png'}, {save_dir_R / f'R_{k:04d}.png'}")
