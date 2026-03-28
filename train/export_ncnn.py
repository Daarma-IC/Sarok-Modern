import argparse
import shutil
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
    exit(1)


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8 detection model to NCNN")
    parser.add_argument("--model", type=str, default=None, help="Path to .pt model")
    parser.add_argument("--imgsz", type=int, default=640,  help="Image size")
    args = parser.parse_args()

    project_dir = Path(__file__).parent.parent
    models_dir  = project_dir / "models"
    model_path  = Path(args.model) if args.model else models_dir / "trash_det_best.pt"

    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        print(f"[TIP]   Run: python train/train.py")
        return

    print("=" * 60)
    print("  YOLOv8 Detection -> NCNN Export")
    print("=" * 60)
    print(f"  Model   : {model_path}")
    print(f"  Img Size: {args.imgsz}")
    print("=" * 60)

    model = YOLO(str(model_path))

    print("\n[INFO] Exporting to NCNN...")
    ncnn_dir  = model.export(format="ncnn", imgsz=args.imgsz)
    ncnn_path = Path(ncnn_dir)

    if ncnn_path.is_dir():
        dest = models_dir / "trash_det_ncnn"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(ncnn_path, dest)
        print(f"  Copied to: {dest}")
        print(f"\n[INFO] NCNN files:")
        for f in sorted(dest.iterdir()):
            print(f"  - {f.name}  ({f.stat().st_size / 1024:.1f} KB)")

    print(f"\n[SUCCESS] Export complete!")
    print(f"[INFO] Copy models/trash_det_ncnn/ to Raspberry Pi.")
    print(f"[NEXT] python inference/detect_camera.py")


if __name__ == "__main__":
    main()
