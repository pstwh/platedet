import argparse
import os
from typing import Any, Dict

import cv2
import numpy as np
from PIL import Image

from platedet import Platedet


def get_args():
    parser = argparse.ArgumentParser(description="Detect license plates in images.")

    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the image you want to detect license plates in.",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "artifacts", "platedet.onnx"),
        help="Path to the ONNX model (default: artifacts/platedet.onnx).",
    )

    parser.add_argument(
        "--return_types",
        type=str,
        nargs="+",
        default=["pil"],
        choices=["raw", "pil", "np_array", "boxes"],
        help="Return types for inference.",
    )

    parser.add_argument(
        "--providers",
        type=str,
        nargs="+",
        default=["CPUExecutionProvider"],
        help="Execution provider for ONNX Runtime (default: CPUExecutionProvider).",
    )

    parser.add_argument(
        "--save_images", action="store_true", help="Save cropped images."
    )
    parser.add_argument(
        "--show_result", action="store_true", help="Display results with boxes."
    )

    return parser.parse_args()


def save_crops(output: Dict[str, Any], image_path: str):
    for idx, crop_info in enumerate(output["pil"]["images"]):
        save_path = image_path.replace(
            os.path.splitext(image_path)[-1], f"_{idx}_plate.jpg"
        )
        crop_info.save(save_path)
        print(f"Saved cropped image: {save_path}")


def show_boxes(image_path: str, output: Dict[str, Any]):
    image = cv2.imread(image_path)
    boxes = output["boxes"]["boxes"]
    confidences = output["boxes"]["confidences"]

    for box, conf in zip(boxes, confidences):
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (127, 255, 127), 2)
        cv2.putText(
            image,
            f"{conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (127, 255, 127),
            2,
        )

    cv2.imshow("Detected Plates", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    args = get_args()

    image = Image.open(args.image_path).convert("RGB")

    return_types = args.return_types
    if args.show_result:
        return_types.append("boxes")
    
    if args.save_images:
        return_types.append("pil")

    platedet = Platedet(model_path=args.model_path, providers=args.providers)
    output = platedet.inference(np.array(image), return_types=args.return_types)

    if args.save_images:
        save_crops(output, args.image_path)

    if args.show_result:
        show_boxes(args.image_path, output)


if __name__ == "__main__":
    main()
