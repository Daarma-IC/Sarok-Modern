"""
Real-time trash detection — YOLOv8 with TACO 60-class model.

Camera overlay: shows specific object name (e.g. "Clear plastic bottle")
Bounding box color: based on category (organik/anorganik/residu/B3)
Terminal log: "[DETECT] Clear plastic bottle -> ANORGANIK (85%)"

Usage:
    python inference/detect_camera.py
    python inference/detect_camera.py --camera 1   # DroidCam
    python inference/detect_camera.py --conf 0.3
    python inference/detect_camera.py --imgsz 320  # lebih cepat

Controls:
    q / ESC  -> Quit
    s        -> Save screenshot
    p        -> Pause / Resume
"""

import cv2
import argparse
import time
from pathlib import Path
from datetime import datetime

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
    exit(1)


# ============================================================
#  Category mapping: TACO class name -> (category, color BGR)
# ============================================================
ORGANIK_COLOR   = (50,  200,  50)   # green
ANORGANIK_COLOR = (50,  165, 255)   # orange
RESIDU_COLOR    = (180, 180, 180)   # gray
B3_COLOR        = (0,    0,  255)   # red

TACO_TO_CATEGORY = {
    # ---- ORGANIK ----
    "Food waste":                   ("ORGANIK",    ORGANIK_COLOR),

    # ---- ANORGANIK ----
    "Aluminium foil":               ("ANORGANIK",  ANORGANIK_COLOR),
    "Aluminium blister pack":       ("ANORGANIK",  ANORGANIK_COLOR),
    "Carded blister pack":          ("ANORGANIK",  ANORGANIK_COLOR),
    "Other plastic bottle":         ("ANORGANIK",  ANORGANIK_COLOR),
    "Clear plastic bottle":         ("ANORGANIK",  ANORGANIK_COLOR),
    "Glass bottle":                 ("ANORGANIK",  ANORGANIK_COLOR),
    "Plastic bottle cap":           ("ANORGANIK",  ANORGANIK_COLOR),
    "Metal bottle cap":             ("ANORGANIK",  ANORGANIK_COLOR),
    "Broken glass":                 ("ANORGANIK",  ANORGANIK_COLOR),
    "Food Can":                     ("ANORGANIK",  ANORGANIK_COLOR),
    "Drink can":                    ("ANORGANIK",  ANORGANIK_COLOR),
    "Toilet tube":                  ("ANORGANIK",  ANORGANIK_COLOR),
    "Other carton":                 ("ANORGANIK",  ANORGANIK_COLOR),
    "Egg carton":                   ("ANORGANIK",  ANORGANIK_COLOR),
    "Drink carton":                 ("ANORGANIK",  ANORGANIK_COLOR),
    "Corrugated carton":            ("ANORGANIK",  ANORGANIK_COLOR),
    "Meal carton":                  ("ANORGANIK",  ANORGANIK_COLOR),
    "Pizza box":                    ("ANORGANIK",  ANORGANIK_COLOR),
    "Paper cup":                    ("ANORGANIK",  ANORGANIK_COLOR),
    "Disposable plastic cup":       ("ANORGANIK",  ANORGANIK_COLOR),
    "Foam cup":                     ("ANORGANIK",  ANORGANIK_COLOR),
    "Glass cup":                    ("ANORGANIK",  ANORGANIK_COLOR),
    "Other plastic cup":            ("ANORGANIK",  ANORGANIK_COLOR),
    "Glass jar":                    ("ANORGANIK",  ANORGANIK_COLOR),
    "Plastic lid":                  ("ANORGANIK",  ANORGANIK_COLOR),
    "Metal lid":                    ("ANORGANIK",  ANORGANIK_COLOR),
    "Other plastic":                ("ANORGANIK",  ANORGANIK_COLOR),
    "Magazine paper":               ("ANORGANIK",  ANORGANIK_COLOR),
    "Tissues":                      ("ANORGANIK",  ANORGANIK_COLOR),
    "Wrapping paper":               ("ANORGANIK",  ANORGANIK_COLOR),
    "Normal paper":                 ("ANORGANIK",  ANORGANIK_COLOR),
    "Paper bag":                    ("ANORGANIK",  ANORGANIK_COLOR),
    "Plastified paper bag":         ("ANORGANIK",  ANORGANIK_COLOR),
    "Plastic film":                 ("ANORGANIK",  ANORGANIK_COLOR),
    "Six pack rings":               ("ANORGANIK",  ANORGANIK_COLOR),
    "Garbage bag":                  ("ANORGANIK",  ANORGANIK_COLOR),
    "Other plastic wrapper":        ("ANORGANIK",  ANORGANIK_COLOR),
    "Single-use carrier bag":       ("ANORGANIK",  ANORGANIK_COLOR),
    "Polypropylene bag":            ("ANORGANIK",  ANORGANIK_COLOR),
    "Crisp packet":                 ("ANORGANIK",  ANORGANIK_COLOR),
    "Spread tub":                   ("ANORGANIK",  ANORGANIK_COLOR),
    "Tupperware":                   ("ANORGANIK",  ANORGANIK_COLOR),
    "Disposable food container":    ("ANORGANIK",  ANORGANIK_COLOR),
    "Foam food container":          ("ANORGANIK",  ANORGANIK_COLOR),
    "Other plastic container":      ("ANORGANIK",  ANORGANIK_COLOR),
    "Plastic glooves":              ("ANORGANIK",  ANORGANIK_COLOR),
    "Plastic utensils":             ("ANORGANIK",  ANORGANIK_COLOR),
    "Pop tab":                      ("ANORGANIK",  ANORGANIK_COLOR),
    "Squeezable tube":              ("ANORGANIK",  ANORGANIK_COLOR),
    "Plastic straw":                ("ANORGANIK",  ANORGANIK_COLOR),
    "Paper straw":                  ("ANORGANIK",  ANORGANIK_COLOR),
    "Styrofoam piece":              ("ANORGANIK",  ANORGANIK_COLOR),

    # ---- RESIDU ----
    "Rope & strings":               ("RESIDU",     RESIDU_COLOR),
    "Scrap metal":                  ("RESIDU",     RESIDU_COLOR),
    "Shoe":                         ("RESIDU",     RESIDU_COLOR),
    "Unlabeled litter":             ("RESIDU",     RESIDU_COLOR),
    "Cigarette":                    ("RESIDU",     RESIDU_COLOR),

    # ---- B3 ----
    "Battery":                      ("B3",         B3_COLOR),
    "Aerosol":                      ("B3",         B3_COLOR),
}

DEFAULT_CATEGORY = ("RESIDU", RESIDU_COLOR)


def get_category(cls_name: str):
    """Return (category_label, color) for a given TACO class name."""
    return TACO_TO_CATEGORY.get(cls_name, DEFAULT_CATEGORY)


def draw_detections(frame, results, conf_threshold):
    detections = []
    if not results or results[0].boxes is None:
        return frame, detections

    names = results[0].names

    for box in results[0].boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue

        cls_id   = int(box.cls[0])
        cls_name = names.get(cls_id, str(cls_id))
        category, color = get_category(cls_name)

        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

        # Line 1: specific object name
        label_obj = f"{cls_name} {conf:.0%}"
        # Line 2: category badge
        label_cat = f"[{category}]"

        # Bounding box in category color
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Object name background
        (tw, th), _ = cv2.getTextSize(label_obj, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1 - th - 28), (x1 + max(tw, 80) + 8, y1), (30, 30, 30), -1)
        cv2.putText(frame, label_obj, (x1 + 4, y1 - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        # Category badge
        cv2.putText(frame, label_cat, (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        detections.append({
            "cls":      cls_name,
            "category": category,
            "conf":     conf,
            "bbox":     (x1, y1, x2, y2),
        })

    return frame, detections


def draw_hud(frame, fps, det_count, paused):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 45), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 0), 2)
    cv2.putText(frame, f"Detections: {det_count}", (160, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    status = "PAUSED" if paused else "LIVE"
    color  = (0, 60, 255) if paused else (0, 220, 0)
    cv2.putText(frame, status, (w - 110, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    # Category legend
    legend = [("ORGANIK", ORGANIK_COLOR), ("ANORGANIK", ANORGANIK_COLOR),
              ("RESIDU", RESIDU_COLOR), ("B3", B3_COLOR)]
    lx = 10
    for name, col in legend:
        cv2.circle(frame, (lx + 6, h - 14), 6, col, -1)
        cv2.putText(frame, name, (lx + 16, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
        lx += 110

    cv2.putText(frame, "[Q]uit [S]save [P]ause", (w - 230, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Real-time trash detection with category mapping")
    parser.add_argument("--model",    type=str,   default=None,  help="Path to .pt or NCNN model folder")
    parser.add_argument("--camera",   type=str,   default="0",   help="Camera index or URL")
    parser.add_argument("--conf",     type=float, default=0.35,  help="Confidence threshold (default: 0.35)")
    parser.add_argument("--imgsz",    type=int,   default=640,   help="Inference image size (default: 640)")
    parser.add_argument("--save-dir", type=str,   default=None,  help="Screenshot save directory")
    args = parser.parse_args()

    project_dir = Path(__file__).parent.parent
    model_path  = Path(args.model) if args.model else project_dir / "models" / "trash_det_best.pt"

    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        print(f"[TIP]   Train: python train/train.py --data dataset/taco_yolo/data.yaml --epochs 50")
        return

    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(str(model_path))
    print(f"[INFO] {len(model.names)} classes loaded")

    try:
        cam_src = int(args.camera)
    except ValueError:
        cam_src = args.camera

    print(f"[INFO] Opening camera: {cam_src}")
    cap = cv2.VideoCapture(cam_src)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera: {cam_src}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    save_dir = Path(args.save_dir) if args.save_dir else project_dir / "screenshots"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 50}")
    print(f"  Trash Detection — LIVE")
    print(f"  Bounding box : specific object name")
    print(f"  Color        : category (organik/anorganik/residu/B3)")
    print(f"  Terminal     : object -> category (conf%)")
    print(f"{'=' * 50}\n")

    paused      = False
    fps         = 0.0
    frame_count = 0
    start_time  = time.time()
    frame       = None
    detections  = []

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                continue

            results = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
            frame, detections = draw_detections(frame, results, args.conf)

            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            if frame_count >= 30:
                frame_count = 0
                start_time  = time.time()

            if detections and frame_count % 10 == 1:
                for d in detections:
                    print(f"[DETECT] {d['cls']} -> {d['category']} ({d['conf']:.0%})")

        if frame is not None:
            display = draw_hud(frame.copy(), fps, len(detections) if not paused else 0, paused)
            cv2.imshow("Trash Detection", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = save_dir / f"det_{ts}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"[SAVE] {path}")
        elif key == ord('p'):
            paused = not paused
            print(f"[INFO] {'Paused' if paused else 'Resumed'}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] Stopped.")


if __name__ == "__main__":
    main()
