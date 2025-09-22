from pathlib import Path

BASE_DIR = Path(__file__).parent

# --- file path ------

NPZ_PATH   = BASE_DIR / "calib_out" / "old_camera_same" / "stereo" / "stereo_params_scaled.npz"
MODEL_PATH = BASE_DIR / "models" / "best_6.pt"

# --- camera params ---
CAM1_INDEX = 1
CAM2_INDEX = 2

# 입력(캡처) 좌/우가 보정(P1/P2)과 뒤집혔다면 True로 (정석 해결)
SWAP_INPUT     = False

# --- UI / display ---
WINDOW_NAME    = "Rectified L | R  (10f merged, LEFT-origin O; MP Left, Auto-ID0)"
SHOW_GRID      = False
THRESH_MASK    = 0.7
ROW_TOL_Y      = 30
SELECTED_COLOR = "green"    # 예: 'orange' (None이면 콘솔 입력/엔터=전체)
SAVE_VIDEO     = False
OUT_FPS        = 30
OUT_PATH       = BASE_DIR / "stereo_overlay.mp4"

CSV_GRIPS_PATH = BASE_DIR / "output_grips.csv"
TOUCH_THRESH = 10         

# ---- 레이저 원점(=조준 기준점) 오프셋 (LEFT 카메라 원점 기준, cm) ----
# 실측: 왼쪽 카메라 중심 기준 왼쪽 1.85cm, 위 8cm, 카메라보다 3.3cm 뒤
LASER_OFFSET_CM_LEFT = 1.85   # '왼쪽'은 x 음(-) 처리
LASER_OFFSET_CM_UP   = 8.0    # '위쪽'은 y 음(-) 처리
LASER_OFFSET_CM_FWD  = -3.3   # 전방 +, 뒤쪽 - → 뒤 3.3cm 이므로 -3.3
Y_UP_IS_NEGATIVE = True       # 위가 -y

# ---- “ID0로 조준”을 실제로 보낼지 옵션 ----
SEND_SERIAL      = False           # True로 바꾸면 시리얼 송신
SERIAL_PORT      = "COM6"          # 보드 포트
SERIAL_BAUD      = 115200

# 간단 오프셋 보정(현장 튜닝)
YAW_OFFSET_DEG   = 0.0
PITCH_OFFSET_DEG = 0.0

# (선택) 2x2 선형 보정 모델 사용 여부
USE_LINEAR_CAL = False
A11, A12, B1 = 1.0, 0.0, 0.0    # yaw_cmd = A11*yaw_est + A12*pitch_est + B1
A21, A22, B2 = 0.0, 1.0, 0.0    # pitch_cmd = A21*yaw_est + A22*pitch_est + B2

# (선택) 서보 각→PWM(us) 맵 — 하드웨어에 맞게 수정
SERVO = {
    "YAW_MIN_DEG":   -90.0, "YAW_MAX_DEG":   90.0, "YAW_MIN_US": 1000, "YAW_MAX_US": 2000,
    "PITCH_MIN_DEG": -45.0, "PITCH_MAX_DEG": 45.0, "PITCH_MIN_US":1000, "PITCH_MAX_US":2000,
}

# (선택) 프리뷰 최대 폭
PREVIEW_MAX_W = None  # 예: 1280

# ==== 초기 YOLO 프레임 수 & 병합 기준 ====
INIT_DET_FRAMES   = 10          # ✅ 첫 10프레임 사용
CENTER_MERGE_PX   = 18          # ✅ 프레임 간 동일 홀드로 간주할 중심거리(px)                      # 연속 프레임 수(>= 이면 채색)


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