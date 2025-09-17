#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stereo YOLOv8n-Seg (first 10 frames merged) + MediaPipe-on-Left → Live overlay (mm)
+ Laser-origin yaw/pitch per hold (LEFT-camera-based)
+ ✅ DualServoController 연동: 초기 각도는 CLI로 1회만 설정, 이후 손가락이 잡으면 다음 홀드로 자동 이동

- 시작 시 첫 10프레임에서 YOLO 세그 → 프레임 간 중복 병합 → y행/x정렬로 hold_index 부여
- 좌/우 공통 hold_index 쌍만 삼각측량 → X(mm), yaw/pitch(레이저 원점=LEFT기준) 계산
- 시작 시: (옵션) --center 또는 --pitch/--yaw 로 1회만 수동 초기세팅
- 라이브: MediaPipe로 '현재 타깃 홀드'에 손가락이 TOUCH_THRESHOLD 프레임 이상 들어오면 다음 홀드로 자동 이동
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
import argparse

# ======== (NEW) DualServoController 가져오기 ========
# - servo_control.py의 직렬 명령 형식을 그대로 사용
# - 없으면 더미 컨트롤러로 안전 동작
try:
    from servo_control import DualServoController  # S pitch yaw / P / Y / C / R / Z
    HAS_SERVO = True
except Exception:
    HAS_SERVO = False
    class DualServoController:
        def __init__(self, *a, **k): print("[Servo] (stub) controller unavailable")
        def set_angles(self, pitch=None, yaw=None): print(f"[Servo] (stub) set_angles: P={pitch}, Y={yaw}")
        def center(self): print("[Servo] (stub) center")
        def query(self): print("[Servo] (stub) query"); return ""
        def laser_on(self): print("[Servo] (stub) laser_on")
        def laser_off(self): print("[Servo] (stub) laser_off")
        def close(self): pass

# ========= 사용자 설정 =========
NPZ_PATH       = r"C:\Users\user\Documents\캡스턴 디자인\triangulation\calib_out\old_camera_same\stereo\stereo_params_scaled.npz"
MODEL_PATH     = r"C:\Users\user\Documents\캡스턴 디자인\triangulation\best_6.pt"

CAM1_INDEX     = 1   # 물리 카메라 인덱스(왼쪽)
CAM2_INDEX     = 2   # 물리 카메라 인덱스(오른쪽)

# 입력(캡처) 좌/우가 보정(P1/P2)과 뒤집혔다면 True로 (정석 해결)
SWAP_INPUT     = False

# 화면(UI)만 좌/우 바꿔서 표시할지 (오버레이/텍스트 오프셋 자동 정합)
SWAP_DISPLAY   = False

WINDOW_NAME    = "Rectified L | R  (10f merged; MP Left; Servo Auto-Advance)"
SHOW_GRID      = False
THRESH_MASK    = 0.7
ROW_TOL_Y      = 30
SELECTED_COLOR = None    # 예: 'orange' (None이면 콘솔 입력/엔터=전체)

SAVE_VIDEO     = False
OUT_FPS        = 30
OUT_PATH       = "stereo_overlay.mp4"

CSV_GRIPS_PATH  = "grip_records.csv"
TOUCH_THRESHOLD = 10  # 연속 프레임
# =================================

# ---- 레이저 원점(=조준 기준점) 오프셋 (LEFT 카메라 원점 기준, cm) ----
LASER_OFFSET_CM_LEFT = 1.85   # '왼쪽'은 x 음(-)
LASER_OFFSET_CM_UP   = 8.0    # '위쪽'은 y 음(-)
LASER_OFFSET_CM_FWD  = -3.3   # 전방 +, 뒤쪽 - → 뒤 3.3cm이므로 -3.3
Y_UP_IS_NEGATIVE = True       # 위가 -y

# 간단 오프셋 보정(현장 튜닝)
YAW_OFFSET_DEG   = 0.0
PITCH_OFFSET_DEG = 0.0

# (선택) 2x2 선형 보정 모델 사용 여부
USE_LINEAR_CAL = False
A11, A12, B1 = 1.0, 0.0, 0.0    # yaw_cmd = A11*yaw_est + A12*pitch_est + B1
A21, A22, B2 = 0.0, 1.0, 0.0    # pitch_cmd = A21*yaw_est + A22*pitch_est + B2

# (선택) 프리뷰 최대 폭
PREVIEW_MAX_W = None  # 예: 1280

# ==== 초기 YOLO 프레임 수 & 병합 기준 ====
INIT_DET_FRAMES   = 10
CENTER_MERGE_PX   = 18
# ==============================

# YOLO 클래스 컬러 (BGR)
COLOR_MAP = {
    'Hold_Red':(0,0,255),'Hold_Orange':(0,165,255),'Hold_Yellow':(0,255,255),
    'Hold_Green':(0,255,0),'Hold_Blue':(255,0,0),'Hold_Purple':(204,50,153),
    'Hold_Pink':(203,192,255),'Hold_Lime':(50,255,128),'Hold_Sky':(255,255,0),
    'Hold_White':(255,255,255),'Hold_Black':(30,30,30),'Hold_Gray':(150,150,150),
}
ALL_COLORS = {
    'red':'Hold_Red','orange':'Hold_Orange','yellow':'Hold_Yellow','green':'Hold_Green',
    'blue':'Hold_Blue','purple':'Hold_Purple','pink':'Hold_Pink','white':'Hold_White',
    'black':'Hold_Black','gray':'Hold_Gray','lime':'Hold_Lime','sky':'Hold_Sky',
}

# ---------- 유틸 ----------
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

def open_cams(idx1, idx2, size):
    W, H = size
    cap1 = cv2.VideoCapture(idx1, cv2.CAP_DSHOW)
    cap2 = cv2.VideoCapture(idx2, cv2.CAP_DSHOW)
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH,  W); cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH,  W); cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    if not cap1.isOpened() or not cap2.isOpened():
        raise SystemExit("카메라를 열 수 없습니다. 인덱스/연결 확인.")
    return cap1, cap2

def rectify(frame, mx, my, size):
    W, H = size
    if (frame.shape[1], frame.shape[0]) != (W, H):
        frame = cv2.resize(frame, (W, H))
    return cv2.remap(frame, mx, my, cv2.INTER_LINEAR)

def extract_holds_with_indices(frame_bgr, model, selected_class_name=None,
                               mask_thresh=0.7, row_tol=50):
    h, w = frame_bgr.shape[:2]
    res = model(frame_bgr)[0]
    holds = []
    if res.masks is None: return []
    masks = res.masks.data; boxes = res.boxes; names = model.names
    for i in range(masks.shape[0]):
        mask = masks[i].cpu().numpy()
        mask_rs = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        binary = (mask_rs > mask_thresh).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
        contour = max(contours, key=cv2.contourArea)
        cls_id = int(boxes.cls[i].item()); conf = float(boxes.conf[i].item())
        class_name = names[cls_id]
        if (selected_class_name is not None) and (class_name != selected_class_name):
            continue
        Mom = cv2.moments(contour)
        if Mom["m00"] == 0: continue
        cx = int(Mom["m10"]/Mom["m00"]); cy = int(Mom["m01"]/Mom["m00"])
        holds.append({"class_name": class_name, "color": COLOR_MAP.get(class_name,(255,255,255)),
                      "contour": contour, "center": (cx, cy), "conf": conf})
    if not holds: return []
    enriched = [{"cx": h_["center"][0], "cy": h_["center"][1], **h_} for h_ in holds]
    enriched.sort(key=lambda h: h["cy"])
    rows, cur = [], [enriched[0]]
    for h_ in enriched[1:]:
        if abs(h_["cy"] - cur[0]["cy"]) < row_tol: cur.append(h_)
        else: rows.append(cur); cur = [h_]
    rows.append(cur)
    final_sorted = []
    for row in rows:
        row.sort(key=lambda h: h["cx"])
        final_sorted.extend(row)
    for idx, h_ in enumerate(final_sorted):
        h_["hold_index"] = idx
    return final_sorted

def merge_holds_by_center(holds_lists, merge_dist_px=18):
    merged = []
    for holds in holds_lists:
        for h in holds:
            h = {k: v for k, v in h.items()}
            h.pop("hold_index", None)
            assigned = False
            for m in merged:
                dx = h["center"][0] - m["center"][0]
                dy = h["center"][1] - m["center"][1]
                if (dx*dx + dy*dy) ** 0.5 <= merge_dist_px:
                    area_h = cv2.contourArea(h["contour"])
                    area_m = cv2.contourArea(m["contour"])
                    if (area_h > area_m) or (abs(area_h - area_m) < 1e-6 and h.get("conf",0) > m.get("conf",0)):
                        m.update(h)
                    assigned = True
                    break
            if not assigned:
                merged.append(h)
    return merged

def assign_indices(holds, row_tol=50):
    if not holds:
        return []
    enriched = [{"cx": h["center"][0], "cy": h["center"][1], **h} for h in holds]
    enriched.sort(key=lambda h: h["cy"])
    rows, cur = [], [enriched[0]]
    for h_ in enriched[1:]:
        if abs(h_["cy"] - cur[0]["cy"]) < row_tol: cur.append(h_)
        else: rows.append(cur); cur = [h_]
    rows.append(cur)
    final_sorted = []
    for row in rows:
        row.sort(key=lambda h: h["cx"])
        final_sorted.extend(row)
    for idx, h_ in enumerate(final_sorted):
        h_["hold_index"] = idx
    return final_sorted

def triangulate_xy(P1, P2, ptL, ptR):
    xl = np.array(ptL, dtype=np.float64).reshape(2,1)
    xr = np.array(ptR, dtype=np.float64).reshape(2,1)
    Xh = cv2.triangulatePoints(P1, P2, xl, xr)
    X  = (Xh[:3] / Xh[3]).reshape(3)  # [X,Y,Z] (mm)
    return X

def draw_grid(img):
    h, w = img.shape[:2]; step = max(20, h//20)
    for y in range(0, h, step):
        cv2.line(img, (0,y), (w-1,y), (0,255,0), 1, cv2.LINE_AA)

def yaw_pitch_from_X(X, O, y_up_is_negative=True):
    v = X - O
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    yaw   = np.degrees(np.arctan2(vx, vz))
    pitch = np.degrees(np.arctan2((-vy if y_up_is_negative else vy), np.hypot(vx, vz)))
    return yaw, pitch

def angle_between(v1, v2):
    a = np.linalg.norm(v1); b = np.linalg.norm(v2)
    if a == 0 or b == 0: return 0.0
    cosang = np.clip(np.dot(v1, v2) / (a * b), -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

def wrap_deg(d): return (d + 180.0) % 360.0 - 180.0

def imshow_scaled(win, img, maxw=None):
    if not maxw: cv2.imshow(win, img); return
    h, w = img.shape[:2]
    if w > maxw:
        s = maxw / w
        img = cv2.resize(img, (int(w*s), int(h*s)))
    cv2.imshow(win, img)

def xoff_for(side, W, swap):
    if side == "L":
        return (W if swap else 0)
    else:
        return (0 if swap else W)

# ---------- (NEW) Servo 보정/전송 ----------
def apply_calibration(yaw_est, pitch_est):
    if USE_LINEAR_CAL:
        yaw_cmd   = A11*yaw_est + A12*pitch_est + B1
        pitch_cmd = A21*yaw_est + A22*pitch_est + B2
    else:
        yaw_cmd   = yaw_est   + YAW_OFFSET_DEG
        pitch_cmd = pitch_est + PITCH_OFFSET_DEG
    return yaw_cmd, pitch_cmd

def send_servo_angles(ctl, yaw_cmd, pitch_cmd):
    # DualServoController는 (pitch, yaw) 순서로 전송
    try:
        print(f"[Servo] send: yaw={yaw_cmd:.2f}°, pitch={pitch_cmd:.2f}°")
        ctl.set_angles(pitch_cmd, yaw_cmd)
    except Exception as e:
        print(f"[Servo ERROR] {e}")

# ---------- 메인 ----------
def main():
    # ---- CLI 인자 (초기 1회 세팅만) ----
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM4", help="서보 보드 포트 (예: COM4)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--center", action="store_true", help="시작 시 1회 센터 이동")
    ap.add_argument("--pitch", type=float, help="시작 시 1회 수동 pitch 각도")
    ap.add_argument("--yaw",   type=float, help="시작 시 1회 수동 yaw 각도")
    ap.add_argument("--laser_on",  action="store_true", help="시작 시 레이저 ON")
    ap.add_argument("--laser_off", action="store_true", help="시작 시 레이저 OFF")
    ap.add_argument("--no_auto_advance", action="store_true", help="손 인식 자동 넘김 비활성화")
    args = ap.parse_args()

    # 경로 검사
    for p in (NPZ_PATH, MODEL_PATH):
        if not Path(p).exists():
            raise FileNotFoundError(f"파일이 없습니다: {p}")

    # 준비
    map1x, map1y, map2x, map2y, P1, P2, size, B, M = load_stereo(NPZ_PATH)
    W, H = size
    print(f"[Info] image_size={(W,H)}, baseline~{B:.2f} mm")

    # 레이저 원점 O (LEFT 카메라 기준 오프셋)
    L = np.array([0.0, 0.0, 0.0], dtype=np.float64)
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
            selected_class_name = ask_color_and_map_to_class(ALL_COLORS)
        else:
            print(f"[Filter] 선택 클래스(상수): {selected_class_name}")
    else:
        selected_class_name = ask_color_and_map_to_class(ALL_COLORS)

    # 카메라 & 모델
    capL_idx, capR_idx = CAM1_INDEX, CAM2_INDEX
    if SWAP_INPUT:
        capL_idx, capR_idx = capR_idx, capL_idx
    cap1, cap2 = open_cams(capL_idx, capR_idx, size)
    model = YOLO(str(MODEL_PATH))

    # ====== 초기 10프레임 수집 & YOLO → 병합 ======
    print(f"[Init] First {INIT_DET_FRAMES} frames: YOLO seg & merge ...")
    L_sets, R_sets = [], []
    for _ in range(2):  # 워밍업
        cap1.read(); cap2.read()
    for k in range(INIT_DET_FRAMES):
        ok1, f1 = cap1.read(); ok2, f2 = cap2.read()
        if not (ok1 and ok2):
            cap1.release(); cap2.release()
            raise SystemExit("초기 프레임 캡처 실패")
        Lr_k = rectify(f1, map1x, map1y, size)
        Rr_k = rectify(f2, map2x, map2y, size)
        holdsL_k = extract_holds_with_indices(Lr_k, model, selected_class_name, THRESH_MASK, ROW_TOL_Y)
        holdsR_k = extract_holds_with_indices(Rr_k, model, selected_class_name, THRESH_MASK, ROW_TOL_Y)
        L_sets.append(holdsL_k); R_sets.append(holdsR_k)
        print(f"  - frame {k+1}/{INIT_DET_FRAMES}: L={len(holdsL_k)}  R={len(holdsR_k)}")

    # 병합 후 인덱스 재부여
    holdsL = assign_indices(merge_holds_by_center(L_sets, CENTER_MERGE_PX), ROW_TOL_Y)
    holdsR = assign_indices(merge_holds_by_center(R_sets, CENTER_MERGE_PX), ROW_TOL_Y)
    if not holdsL or not holdsR:
        cap1.release(); cap2.release()
        print("[Warn] 한쪽 또는 양쪽에서 홀드가 검출되지 않았습니다.")
        return

    # 좌/우 공통 hold_index
    idxL = {h["hold_index"]: h for h in holdsL}
    idxR = {h["hold_index"]: h for h in holdsR}
    common_ids = sorted(set(idxL.keys()) & set(idxR.keys()))
    if not common_ids:
        print("[Warn] 좌/우 공통 hold_index가 없습니다.")
    else:
        print(f"[Info] 매칭된 홀드 쌍 수: {len(common_ids)}")

    # 매칭 결과(3D/각도) — LEFT 원점 기반
    matched_results = []
    for hid in common_ids:
        Lh = idxL[hid]; Rh = idxR[hid]
        X = triangulate_xy(P1, P2, Lh["center"], Rh["center"])
        yaw_deg, pitch_deg = yaw_pitch_from_X(X, O, Y_UP_IS_NEGATIVE)
        matched_results.append({
            "hid": hid, "color": Lh["color"], "X": X,
            "yaw_deg": yaw_deg, "pitch_deg": pitch_deg,
        })
    by_id = {mr["hid"]: mr for mr in matched_results}
    sorted_ids = sorted(by_id.keys())

    # ===== (NEW) 서보 컨트롤러 초기화 & 1회 초기세팅 =====
    ctl = DualServoController(args.port, args.baud) if HAS_SERVO else DualServoController()
    try:
        if args.center:
            print(ctl.center())
        if args.laser_on:
            ctl.laser_on()
        if args.laser_off:
            ctl.laser_off()
        if (args.pitch is not None) or (args.yaw is not None):
            # ✅ 사용자가 CLI로 준 초기 각도 1회만 적용
            print(ctl.set_angles(args.pitch, args.yaw))
        else:
            # 사용자가 초기각도 미지정이면, 첫 타깃(최소 ID) 각도로 1회 이동
            if sorted_ids:
                first = by_id[sorted_ids[0]]
                yaw_cmd, pitch_cmd = apply_calibration(first["yaw_deg"], first["pitch_deg"])
                send_servo_angles(ctl, yaw_cmd, pitch_cmd)
    except Exception as e:
        print(f"[Servo Init ERROR] {e}")

    # ==== MediaPipe Pose (왼쪽 카메라 전용) ====
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
    important_landmarks = {"left_index": 15, "right_index": 16}
    hand_parts = set(important_landmarks.keys())

    # 터치 기록 상태
    grip_records = []         # [part, hold_id, cx, cy]
    already_grabbed = {}      # key=(name, hold_index) → True
    touch_counters = {}       # key=(name, hold_index) → 연속 카운트

    # (NEW) 자동 진행 상태
    auto_advance_enabled = (not args.no_auto_advance)
    current_target_id = sorted_ids[0] if sorted_ids else None
    last_advanced_time = 0.0
    ADV_COOLDOWN = 0.5  # 초과 트리거 방지(초)

    # 비디오 저장
    out = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUT_PATH, fourcc, OUT_FPS, (W*2, H))

    # 라이브 루프
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    t_prev = time.time(); frame_idx = 0

    while True:
        ok1, f1 = cap1.read(); ok2, f2 = cap2.read()
        if not (ok1 and ok2):
            print("[Warn] 프레임 읽기 실패"); break

        Lr = rectify(f1, map1x, map1y, size)
        Rr = rectify(f2, map2x, map2y, size)

        # 화면 결합(표시만 스왑 옵션)
        vis = np.hstack([Rr, Lr]) if SWAP_DISPLAY else np.hstack([Lr, Rr])
        if SHOW_GRID:
            draw_grid(vis[:, :W]); draw_grid(vis[:, W:])

        # 초기 10프레임 병합 결과를 라벨로 그림 (좌/우 둘 다)
        for side, holds in (("L", holdsL), ("R", holdsR)):
            xoff = xoff_for(side, W, SWAP_DISPLAY)
            for h in holds:
                cnt_shifted = h["contour"] + np.array([[[xoff, 0]]], dtype=h["contour"].dtype)
                cv2.drawContours(vis, [cnt_shifted], -1, h["color"], 2)
                cx, cy = h["center"]
                cv2.circle(vis, (cx+xoff, cy), 4, (255,255,255), -1)
                tag = f"ID:{h['hold_index']}"
                if (current_target_id is not None) and (h["hold_index"] == current_target_id):
                    tag = "[TARGET] " + tag
                cv2.putText(vis, tag, (cx+xoff-10, cy+26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(vis, tag, (cx+xoff-10, cy+26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, h["color"], 2, cv2.LINE_AA)

        # 타깃 각도 텍스트
        y0 = 26
        if current_target_id in by_id:
            mr = by_id[current_target_id]
            txt = (f"TARGET ID{mr['hid']}  "
                   f"yaw={mr['yaw_deg']:.1f}°, pitch={mr['pitch_deg']:.1f}°")
            cv2.putText(vis, txt, (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(vis, txt, (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 1, cv2.LINE_AA)
            y0 += 26

        # MediaPipe Pose: 왼쪽만
        image_rgb = cv2.cvtColor(Lr, cv2.COLOR_BGR2RGB)
        result = pose.process(image_rgb)
        pose_landmarks = result.pose_landmarks

        if pose_landmarks:
            hL, wL = Lr.shape[:2]
            coords = {}
            for name, idx in important_landmarks.items():
                lm = pose_landmarks.landmark[idx]
                coords[name] = (lm.x * wL, lm.y * hL)

            left_xoff = xoff_for("L", W, SWAP_DISPLAY)
            for name, (x, y) in coords.items():
                joint_color = (0, 0, 255) if name in hand_parts else (0, 255, 0)
                cv2.circle(vis, (int(x)+left_xoff, int(y)), 6, joint_color, -1)
                cv2.putText(vis, f"{name}:({int(x)},{int(y)})",
                            (int(x)+left_xoff+6, int(y)-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1, cv2.LINE_AA)

            # (NEW) 현재 타깃 홀드에 대한 '손가락 in polygon' 카운트 → 자동 진행
            if auto_advance_enabled and (current_target_id in idxL):
                hold = idxL[current_target_id]
                for name, (x, y) in coords.items():
                    inside = cv2.pointPolygonTest(hold["contour"], (x, y), False) >= 0
                    key = (name, current_target_id)
                    if inside:
                        touch_counters[key] = touch_counters.get(key, 0) + 1
                        if touch_counters[key] >= TOUCH_THRESHOLD:
                            now = time.time()
                            if now - last_advanced_time > ADV_COOLDOWN:
                                # 그립 기록 저장 (1회)
                                if not already_grabbed.get(key):
                                    cx, cy = hold["center"]
                                    grip_records.append([name, current_target_id, cx, cy])
                                    already_grabbed[key] = True

                                # 다음 타깃으로 이동
                                cur_idx = sorted_ids.index(current_target_id) if current_target_id in sorted_ids else -1
                                if (cur_idx >= 0) and (cur_idx + 1 < len(sorted_ids)):
                                    next_id = sorted_ids[cur_idx + 1]
                                    current_target_id = next_id
                                    nxt = by_id[next_id]
                                    yaw_cmd, pitch_cmd = apply_calibration(nxt["yaw_deg"], nxt["pitch_deg"])
                                    send_servo_angles(ctl, yaw_cmd, pitch_cmd)
                                    print(f"[Auto-Advance] → ID{next_id}")
                                    last_advanced_time = now
                                else:
                                    print("[Auto-Advance] 더 이상 다음 홀드가 없습니다.")
                    else:
                        touch_counters[key] = 0

        # FPS
        t_now = time.time(); fps = 1.0 / max(t_now - (t_prev), 1e-6); t_prev = t_now
        cv2.putText(vis, f"FPS: {fps:.1f} (YOLO merged 10f; MP Left; Auto-Advance={'ON' if auto_advance_enabled else 'OFF'})",
                    (10, H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(vis, f"FPS: {fps:.1f} (YOLO merged 10f; MP Left; Auto-Advance={'ON' if auto_advance_enabled else 'OFF'})",
                    (10, H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1, cv2.LINE_AA)

        imshow_scaled(WINDOW_NAME, vis, PREVIEW_MAX_W)
        if SAVE_VIDEO: out.write(vis)

        frame_idx += 1
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        # 수동 스킵: 'n' → 다음 홀드로 강제 이동
        elif k == ord('n') and sorted_ids:
            if current_target_id in sorted_ids:
                i = sorted_ids.index(current_target_id)
                if i + 1 < len(sorted_ids):
                    current_target_id = sorted_ids[i+1]
                    nxt = by_id[current_target_id]
                    yaw_cmd, pitch_cmd = apply_calibration(nxt["yaw_deg"], nxt["pitch_deg"])
                    send_servo_angles(ctl, yaw_cmd, pitch_cmd)
                    print(f"[Manual Next] → ID{current_target_id}")

    # 정리
    cap1.release(); cap2.release()
    if SAVE_VIDEO:
        out.release(); print(f"[Info] 저장 완료: {OUT_PATH}")
    cv2.destroyAllWindows()
    try:
        ctl.close()
    except:
        pass

    # 그립 기록 저장
    with open(CSV_GRIPS_PATH, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["part", "hold_id", "cx", "cy"])
        writer.writerows(grip_records)
    print(f"[Info] 그립 CSV: {CSV_GRIPS_PATH} (행 수: {len(grip_records)})")

if __name__ == "__main__":
    main()
