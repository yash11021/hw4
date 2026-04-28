import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
    """

    with open(info_path) as f:
        info = json.load(f)

    if view_index >= len(info["detections"]):
        return []

    frame_detections = info["detections"][view_index]
    karts = info["karts"]

    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    kart_objects = []
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        x1_s = x1 * scale_x
        y1_s = y1 * scale_y
        x2_s = x2 * scale_x
        y2_s = y2 * scale_y

        if (x2_s - x1_s) < min_box_size or (y2_s - y1_s) < min_box_size:
            continue

        if x2_s < 0 or x1_s > img_width or y2_s < 0 or y1_s > img_height:
            continue

        kart_objects.append({
            "instance_id": track_id,
            "kart_name": karts[track_id] if track_id < len(karts) else f"kart_{track_id}",
            "center": ((x1_s + x2_s) / 2, (y1_s + y2_s) / 2),
            "is_center_kart": False,
        })

    if kart_objects:
        img_cx, img_cy = img_width / 2, img_height / 2
        closest = min(kart_objects, key=lambda k: (k["center"][0] - img_cx) ** 2 + (k["center"][1] - img_cy) ** 2)
        closest["is_center_kart"] = True

    return kart_objects


def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """

    with open(info_path) as f:
        info = json.load(f)
    return info["track"]


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of dictionaries, each containing a question and answer
    """
    with open(info_path) as f:
        info = json.load(f)

    qa_pairs = []
    karts = info["karts"]
    ego_name = karts[view_index]
    distances = info["distance_down_track"]
    ego_dist = distances[view_index]

    # 1. Ego car question
    # What kart is the ego car?
    qa_pairs.append({"question": "What kart is the ego car?", "answer": ego_name})

    # 2. Total karts question
    # How many karts are there in the scenario?
    kart_objects_all = extract_kart_objects(info_path, view_index, img_width, img_height)
    qa_pairs.append({"question": "How many karts are there in the scenario?", "answer": str(len(kart_objects_all))})

    # 3. Track information questions
    # What track is this?
    qa_pairs.append({"question": "What track is this?", "answer": info["track"]})

    # Get visible non-ego karts for positional questions
    img_cx = img_width / 2
    others = [k for k in kart_objects_all if k["instance_id"] != view_index]

    left, right, front, behind = [], [], [], []
    for k in others:
        name = k["kart_name"]
        cx = k["center"][0]
        other_dist = distances[k["instance_id"]]
        h_side = "left" if cx < img_cx else "right"
        v_side = "front" if other_dist > ego_dist else "back"

        # 4. Relative position questions for each kart
        # Is {kart_name} to the left or right of the ego car?
        qa_pairs.append({"question": f"Is {name} to the left or right of the ego car?", "answer": h_side})
        # Is {kart_name} in front of or behind the ego car?
        qa_pairs.append({"question": f"Is {name} in front of or behind the ego car?", "answer": "behind" if v_side == "back" else v_side})
        # Where is {kart_name} relative to the ego car?
        qa_pairs.append({"question": f"Where is {name} relative to the ego car?", "answer": f"{v_side} and {h_side}"})

        (left if h_side == "left" else right).append(name)
        (front if v_side == "front" else behind).append(name)

    # 5. Counting questions
    # How many karts are to the left of the ego car?
    qa_pairs.append({"question": "How many karts are to the left of the ego car?", "answer": str(len(left))})
    # How many karts are to the right of the ego car?
    qa_pairs.append({"question": "How many karts are to the right of the ego car?", "answer": str(len(right))})
    # How many karts are in front of the ego car?
    qa_pairs.append({"question": "How many karts are in front of the ego car?", "answer": str(len(front))})
    # How many karts are behind the ego car?
    qa_pairs.append({"question": "How many karts are behind the ego car?", "answer": str(len(behind))})

    return qa_pairs


def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)


def generate_all(data_dir: str = None, split: str = "train", output_file: str = None):
    """
    Generate QA pairs for all info files in a split and write to a JSON file.

    Args:
        data_dir: Path to the data directory (default: DATA_DIR from data.py)
        split: Dataset split to generate for (default: 'train')
        output_file: Output JSON file path (default: <data_dir>/<split>/balanced_qa_pairs.json)
    """
    from .data import DATA_DIR

    split_dir = Path(data_dir) / split if data_dir else DATA_DIR / split
    out_path = Path(output_file) if output_file else split_dir / "balanced_qa_pairs.json"

    all_pairs = []
    info_files = sorted(split_dir.glob("*_info.json"))
    print(f"Found {len(info_files)} info files in {split_dir}")

    for info_file in info_files:
        base = info_file.stem.replace("_info", "")
        for view_index in range(10):
            img = split_dir / f"{base}_{view_index:02d}_im.jpg"
            if not img.exists():
                continue
            try:
                pairs = generate_qa_pairs(str(info_file), view_index)
                for p in pairs:
                    p["image_file"] = f"{split}/{img.name}"
                all_pairs.extend(pairs)
            except Exception as e:
                print(f"Skipped {info_file.name} view {view_index}: {e}")

    with open(out_path, "w") as f:
        json.dump(all_pairs, f)
    print(f"Generated {len(all_pairs)} pairs -> {out_path}")


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0

Generate all QA pairs for training:
   python -m homework.generate_qa generate_all
"""


def main():
    fire.Fire({"check": check_qa_pairs, "generate_all": generate_all})


if __name__ == "__main__":
    main()
