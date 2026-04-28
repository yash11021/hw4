from pathlib import Path

import fire
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
    import json
    from .generate_qa import extract_kart_objects

    with open(info_path) as f:
        info = json.load(f)

    karts = info["karts"]
    ego_name = karts[view_index]
    distances = info["distance_down_track"]
    ego_dist = distances[view_index]

    kart_objects = extract_kart_objects(info_path, view_index, img_width, img_height)
    others = [k for k in kart_objects if k["instance_id"] != view_index]
    img_cx = img_width / 2

    captions = []

    # 1. Ego car
    # {kart_name} is the ego car.
    captions.append(f"{ego_name} is the ego car.")

    # 2. Counting
    # There are {num_karts} karts in the scenario.
    captions.append(f"There are {len(kart_objects)} karts in the scene.")

    # 3. Track name
    # The track is {track_name}.
    captions.append(f"The track is {info['track']}.")

    # 4. Relative position
    # {kart_name} is {position} of the ego car.
    for k in others:
        name = k["kart_name"]
        cx = k["center"][0]
        other_dist = distances[k["instance_id"]]
        h_side = "left" if cx < img_cx else "right"
        v_side = "in front" if other_dist > ego_dist else "behind"
        captions.append(f"{name} is to the {h_side} and {v_side} of the ego car.")

    return captions


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


def generate_all(data_dir: str = None, split: str = "train", output_file: str = None):
    """
    Generate captions for all info files in a split and write to a JSON file.
    """
    import json
    from .data import DATA_DIR

    split_dir = Path(data_dir) / split if data_dir else DATA_DIR / split
    out_path = Path(output_file) if output_file else split_dir / "captions.json"

    all_captions = []
    info_files = sorted(split_dir.glob("*_info.json"))
    print(f"Found {len(info_files)} info files in {split_dir}")

    for info_file in info_files:
        base = info_file.stem.replace("_info", "")
        for view_index in range(10):
            img = split_dir / f"{base}_{view_index:02d}_im.jpg"
            if not img.exists():
                continue
            try:
                captions = generate_caption(str(info_file), view_index)
                for caption in captions:
                    all_captions.append({
                        "image_file": f"{split}/{img.name}",
                        "caption": caption,
                    })
            except Exception as e:
                print(f"Skipped {info_file.name} view {view_index}: {e}")

    with open(out_path, "w") as f:
        json.dump(all_captions, f)
    print(f"Generated {len(all_captions)} captions -> {out_path}")


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

Generate all captions for training:
   python -m homework.generate_captions generate_all
"""


def main():
    fire.Fire({"check": check_caption, "generate_all": generate_all})


if __name__ == "__main__":
    main()
