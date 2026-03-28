import json
import shutil
import random
from pathlib import Path


TRAIN_RATIO = 0.8


def coco_to_yolo(bbox, img_w, img_h):
    x, y, w, h = bbox
    cx = max(0.0, min(1.0, (x + w / 2) / img_w))
    cy = max(0.0, min(1.0, (y + h / 2) / img_h))
    nw = max(0.0, min(1.0, w / img_w))
    nh = max(0.0, min(1.0, h / img_h))
    return cx, cy, nw, nh


def find_taco_dir():
    project_dir = Path(__file__).parent.parent
    candidates = [
        project_dir / "TACO",
        project_dir / "taco",
        Path("D:/trash_classifier/TACO"),
        Path("D:/trash_classifier/taco"),
    ]
    for p in candidates:
        if (p / "data" / "annotations.json").exists():
            return p
    return None


def main():
    taco_dir = find_taco_dir()
    if taco_dir is None:
        print("[ERROR] TACO not found! Expected: D:/trash_classifier/TACO/data/annotations.json")
        return

    ann_file = taco_dir / "data" / "annotations.json"
    out_dir  = Path(__file__).parent.parent / "dataset" / "taco_yolo"

    print("=" * 60)
    print("  TACO -> YOLOv8 Conversion")
    print(f"  Source: {taco_dir}")
    print("=" * 60)

    with open(ann_file, encoding="utf-8") as f:
        coco = json.load(f)

    images      = {img["id"]: img for img in coco["images"]}
    categories  = {cat["id"]: cat["name"] for cat in coco["categories"]}
    class_names = [categories[k] for k in sorted(categories.keys())]
    cat_to_idx  = {k: i for i, k in enumerate(sorted(categories.keys()))}

    print(f"[INFO] {len(images)} images, {len(coco['annotations'])} annotations, {len(class_names)} classes")
    print(f"[INFO] Keeping all {len(class_names)} original TACO classes")

    ann_by_img = {}
    for ann in coco["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    valid_ids = [
        iid for iid, img in images.items()
        if (taco_dir / "data" / img["file_name"]).exists() and iid in ann_by_img
    ]
    print(f"[INFO] Valid images: {len(valid_ids)}")

    random.seed(42)
    random.shuffle(valid_ids)
    split  = int(len(valid_ids) * TRAIN_RATIO)
    splits = {"train": valid_ids[:split], "val": valid_ids[split:]}

    if out_dir.exists():
        print(f"[INFO] Removing old taco_yolo/...")
        shutil.rmtree(out_dir)

    for s in ("train", "val"):
        (out_dir / "images" / s).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / s).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0}
    for split_name, ids in splits.items():
        for iid in ids:
            img  = images[iid]
            src  = taco_dir / "data" / img["file_name"]
            stem = img["file_name"].replace("/", "_").replace("\\", "_")
            base = Path(stem).stem
            ext  = Path(stem).suffix or ".jpg"

            shutil.copy2(src, out_dir / "images" / split_name / (base + ext))

            with open(out_dir / "labels" / split_name / (base + ".txt"), "w") as lf:
                for ann in ann_by_img[iid]:
                    ci = cat_to_idx[ann["category_id"]]
                    cx, cy, nw, nh = coco_to_yolo(ann["bbox"], img["width"], img["height"])
                    lf.write(f"{ci} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
            counts[split_name] += 1

    yaml_path = out_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {out_dir.resolve()}\n")
        f.write("train: images/train\nval: images/val\n\n")
        f.write(f"nc: {len(class_names)}\n\nnames:\n")
        for i, name in enumerate(class_names):
            f.write(f"  {i}: {name}\n")

    print(f"\n[DONE] train: {counts['train']}  val: {counts['val']}")
    print(f"[YAML] {yaml_path}")
    print(f"\n[NEXT] python train/train.py --data dataset/taco_yolo/data.yaml --epochs 50")


if __name__ == "__main__":
    main()
