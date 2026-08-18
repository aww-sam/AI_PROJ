from pathlib import Path

import yaml
from PIL import Image

# Initial Configuration
ROOT = Path("pcb-defect-dataset")
DATA_YAML = ROOT / "data.yaml"

OUTPUT_DIR=Path("data")
PADDING=0.15 # for extra padding around each 

## to load classes names
def load_class_names() -> list[str]:

    config = yaml.safe_load(DATA_YAML.read_text())
    names = config['names']

    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return list(names)

#crop the image to make it 15% larger
def crop_with_padding(image:Image.Image, x_c, y_c, w, h, pad=PADDING):

    img_w , img_h = image.size
    box_w , box_h = w*img_w , h*img_h
    cx , cy = x_c*img_w , y_c * img_h

    box_w *= 1 + pad
    box_h *= 1 + pad

    left = max(0, cx - box_w / 2)
    top = max(0, cy - box_h / 2)
    right = min(img_w, cx + box_w / 2)
    bottom = min(img_h, cy + box_h / 2)

    return image.crop((left,top,right,bottom))


def process_split(source_split: str,dest_split: str , class_names: str, counters: dict):

    images_dir = ROOT/source_split/"images"
    labels_dir = ROOT/source_split/"labels"

    if not images_dir.exists() or not labels_dir.exists():

        print(f"Skipping {source_split} : {images_dir} or {labels_dir} not found. ")
        return
    
    label_files = sorted(labels_dir.glob("*.txt"))

    for label_file in label_files:
        image_path = None

        for ext in (".jpg", ".jpeg", ".png"):
            candidate = images_dir / (label_file.stem + ext)

            if candidate.exists():
                image_path = candidate
                break

        if image_path is None:
            continue

        image = Image.open(image_path).convert("RGB")
        lines = [l.strip() for l in label_file.read_text().splitlines() if l.strip()]

        for i,line in enumerate(lines):
            parts = line.split()
            class_id = int(parts[0])
            x_c, y_c, w, h = map(float,parts[1:5])
            cls_name = class_names[class_id]

            crop = crop_with_padding(image,x_c,y_c,w,h)
            if crop.size[0] < 10 or crop.size[1] < 10:
                continue ## to skip degenerate crops

            out_path = OUTPUT_DIR/ dest_split / cls_name / f"{source_split}_{label_file.stem}_{i}.jpg"
            crop.save(out_path)
            counters[cls_name] = counters.get(cls_name,0) + 1

def main():
    class_names = load_class_names()
    print("classes:", class_names)

    for split in ("train", "val"):
        for cls in class_names:
            (OUTPUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)

        counters = {cls: 0 for cls in class_names}

        process_split("train", "train", class_names, counters)
        process_split("val", "val", class_names, counters)
        process_split("test", "val", class_names, counters)

    print("Done. Crops per class:")
    for cls, count in counters.items():
        print(f"  {cls}: {count}")
 
 
if __name__ == "__main__":
    main()
            




