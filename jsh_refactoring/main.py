#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stereo YOLOv8n-Seg (first 10 frames merged) + MediaPipe-on-Left → Live overlay (mm)
+ Laser-origin yaw/pitch per hold (LEFT-camera-based) + auto-target ID0 (absolute command)

- 시작 시 첫 10프레임에서 YOLO 세그 → 프레임 간 중복 병합 → y행/x정렬로 hold_index 부여
- 좌/우 공통 hold_index 쌍만 삼각측량 → X(mm), |X−L|, d_line, yaw/pitch(레이저 원점=LEFT기준) 계산
- 시작 시: ID 0(없으면 가장 작은 ID) 자동 선택 → 명령 각 산출(보정 반영) → (옵션) 시리얼 송신
- 라이브: 레티파이 프레임을 화면에 표시할 때 좌/우 스왑 옵션(SWAP_DISPLAY)으로 UI 정렬
- 저장: grip_records.csv 만 저장
"""

import time
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import mediapipe as mp
import csv
import math
# ------------------------------------------------
from config import *
import stereo_utils as su
# ------------------------------------------------


def main():
    # ----------------경로 검사 ------------------------------
    for p in (NPZ_PATH, MODEL_PATH):
        if not Path(p).exists():
            raise FileNotFoundError(f"파일이 없습니다: {p}")
    print("모든 파일이 존재합니다.\n")
    print(f'NPZ path : {NPZ_PATH}')
    print(f'Model path : {MODEL_PATH}')

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
    # -------------------------------------------------------


if __name__ == "__main__":
    main()