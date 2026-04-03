import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from models import build_model


# Hardcoded config
CHECKPOINT_PATH = Path(r"C:\Users\User\Downloads\CrowdCounting-P2PNet\best_mae.pth")
IMAGE_PATH = Path(r"C:\Users\User\Downloads\CrowdCounting-P2PNet\steelbar_dataset\images\00000815781100048206.Image.002623_p96.jpg")
OUTPUT_IMAGE_PATH = Path(r"C:\Users\User\Downloads\CrowdCounting-P2PNet\inference_result.jpg")
OUTPUT_JSON_PATH = Path(r"C:\Users\User\Downloads\CrowdCounting-P2PNet\inference_result.json")

BACKBONE = "vgg16_bn"
ROW = 2
LINE = 2
THRESHOLD = 0.5
POINT_RADIUS = 4
POINT_COLOR_BGR = (0, 0, 255)
TEXT_COLOR_BGR = (255, 255, 255)
TEXT_BG_COLOR_BGR = (0, 0, 0)


def build_config():
    return SimpleNamespace(
        backbone=BACKBONE,
        row=ROW,
        line=LINE,
    )


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    # Handle checkpoints saved from DataParallel/DDP.
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        cleaned_state_dict[key.replace("module.", "", 1)] = value

    model.load_state_dict(cleaned_state_dict, strict=True)
    return checkpoint


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(image).unsqueeze(0)
    return image, tensor


@torch.no_grad()
def predict_points(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    outputs = model(image_tensor)
    scores = torch.softmax(outputs["pred_logits"], dim=-1)[0, :, 1]
    points = outputs["pred_points"][0]

    keep = scores > THRESHOLD
    kept_scores = scores[keep].detach().cpu().numpy()
    kept_points = points[keep].detach().cpu().numpy()

    return kept_points, kept_scores


def draw_points(image, points, scores):
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    for point, score in zip(points, scores):
        x, y = int(round(point[0])), int(round(point[1]))
        cv2.circle(image_bgr, (x, y), POINT_RADIUS, POINT_COLOR_BGR, -1)
        cv2.putText(
            image_bgr,
            f"{score:.2f}",
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            TEXT_COLOR_BGR,
            1,
            cv2.LINE_AA,
        )

    count_text = f"Count: {len(points)}"
    text_size, _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(image_bgr, (10, 10), (20 + text_size[0], 20 + text_size[1]), TEXT_BG_COLOR_BGR, -1)
    cv2.putText(
        image_bgr,
        count_text,
        (15, 15 + text_size[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        TEXT_COLOR_BGR,
        2,
        cv2.LINE_AA,
    )

    return image_bgr


def save_points(points, scores, output_json_path):
    payload = {
        "count": int(len(points)),
        "threshold": THRESHOLD,
        "points": [
            {"x": float(point[0]), "y": float(point[1]), "score": float(score)}
            for point, score in zip(points, scores)
        ],
    }
    output_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    OUTPUT_IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The first model build will auto-download VGG weights if they are not already cached.
    model = build_model(build_config(), training=False)
    model.to(device)
    load_checkpoint(model, CHECKPOINT_PATH, device)
    model.eval()

    original_image, image_tensor = preprocess_image(IMAGE_PATH)
    points, scores = predict_points(model, image_tensor, device)

    drawn = draw_points(original_image, points, scores)
    cv2.imwrite(str(OUTPUT_IMAGE_PATH), drawn)
    save_points(points, scores, OUTPUT_JSON_PATH)

    print(f"Input image: {IMAGE_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Predicted count: {len(points)}")
    print(f"Saved image: {OUTPUT_IMAGE_PATH}")
    print(f"Saved points: {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()
