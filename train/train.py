import argparse
import shutil
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
    exit(1)


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8n detection for trash")
    parser.add_argument("--data",    type=str,   default=None,         help="Path to data.yaml")
    parser.add_argument("--epochs",  type=int,   default=50,           help="Number of epochs")
    parser.add_argument("--imgsz",   type=int,   default=640,          help="Image size")
    parser.add_argument("--batch",   type=int,   default=16,           help="Batch size")
    parser.add_argument("--weights", type=str,   default="yolov8n.pt", help="Pretrained weights")
    parser.add_argument("--device",  type=str,   default=None,         help="Device: cpu or 0 for GPU")
    parser.add_argument("--name",    type=str,   default="trash_det",  help="Run name")
    args = parser.parse_args()

    project_dir = Path(__file__).parent.parent
    data_yaml   = Path(args.data) if args.data else project_dir / "SAMPAH.v1-data-baru.yolov8" / "data.yaml"
    runs_dir    = project_dir / "runs"
    models_dir  = project_dir / "models"
    models_dir.mkdir(exist_ok=True)

    if not data_yaml.exists():
        print(f"[ERROR] data.yaml not found: {data_yaml}")
        return

    print("=" * 60)
    print("  YOLOv8n Detection — Trash Detector")
    print("=" * 60)
    print(f"  Weights : {args.weights}")
    print(f"  Data    : {data_yaml}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Img Size: {args.imgsz}")
    print(f"  Batch   : {args.batch}")
    print(f"  Device  : {args.device or 'auto'}")
    print("=" * 60)

    model = YOLO(args.weights)

    train_kwargs = dict(
        data     = str(data_yaml),
        epochs   = args.epochs,
        imgsz    = args.imgsz,
        batch    = args.batch,
        project  = str(runs_dir),
        name     = args.name,
        exist_ok = True,
        patience = 15,
    )
    if args.device is not None:
        train_kwargs["device"] = args.device

    print(f"\n[INFO] Starting training...\n")
    model.train(**train_kwargs)

    best_src = runs_dir / args.name / "weights" / "best.pt"
    if best_src.exists():
        best_dst = models_dir / "trash_det_best.pt"
        shutil.copy2(best_src, best_dst)
        print(f"\n[SUCCESS] Best model -> {best_dst}")
    else:
        print(f"\n[WARN] best.pt not found at {best_src}")

    print(f"\n[NEXT STEPS]")
    print(f"  Export : python train/export_ncnn.py")
    print(f"  Camera : python inference/detect_camera.py")


if __name__ == "__main__":
    main()
