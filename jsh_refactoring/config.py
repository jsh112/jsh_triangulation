from pathlib import Path

BASE_DIR = Path(__file__).parent

# --- file path ------

NPZ_PATH   = BASE_DIR / "calib_out" / "old_camera_same" / "stereo" / "stereo_params_scaled.npz"
MODEL_PATH = BASE_DIR / "models" / "best_6.pt"