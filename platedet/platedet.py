import os
from enum import Enum
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


class PlatedetOutputType(Enum):
    RAW = "raw"
    PIL = "pil"
    NP_ARRAY = "np_array"
    BOXES = "boxes"


class Platedet:
    def __init__(
        self,
        model_path: str = os.path.join(
            os.path.dirname(__file__), "artifacts", "platedet.onnx"
        ),
        sess_options=ort.SessionOptions(),
        providers=["CPUExecutionProvider"],
    ):
        self.session = ort.InferenceSession(
            model_path, sess_options=sess_options, providers=providers
        )
        self.model_height, self.model_width = self.session.get_inputs()[0].shape[-2:]

    def extract_roi(
        self,
        im0: np.ndarray,
        conf_threshold: float,
        iou_threshold: float,
        nm: int = 32,
        return_types: List[Union[PlatedetOutputType, str]] = [
            PlatedetOutputType.NP_ARRAY
        ],
    ) -> Dict[str, Any]:
        return_types = self._normalize_return_types(return_types)
        processed_img, ratio, (pad_w, pad_h) = self.preprocess(im0)
        preds = self.session.run(
            None, {self.session.get_inputs()[0].name: processed_img.astype(np.float16)}
        )
        results = self.postprocess(
            preds,
            im0,
            ratio,
            pad_w,
            pad_h,
            conf_threshold,
            iou_threshold,
            nm,
            return_types,
        )
        return results

    def preprocess(
        self, img: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
        shape = img.shape[:2]
        r = min(self.model_height / shape[0], self.model_width / shape[1])
        ratio = (r, r)
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        pad_w, pad_h = (
            (self.model_width - new_unpad[0]) / 2,
            (self.model_height - new_unpad[1]) / 2,
        )

        img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        img_padded = cv2.copyMakeBorder(
            img_resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        img_normalized = (
            np.ascontiguousarray(np.einsum("HWC->CHW", img_padded), dtype=np.float32)
            / 255.0
        )
        img_batch = (
            img_normalized[None] if len(img_normalized.shape) == 3 else img_normalized
        )

        return img_batch, ratio, (pad_w, pad_h)

    def postprocess(
        self,
        preds: Any,
        im0: np.ndarray,
        ratio: Tuple[float, float],
        pad_w: float,
        pad_h: float,
        conf_threshold: float,
        iou_threshold: float,
        nm: int = 32,
        return_types: List[PlatedetOutputType] = [PlatedetOutputType.PIL],
    ) -> Dict[str, Any]:
        x, protos = preds
        x = np.einsum("BCN->BNC", x)
        mask = np.amax(x[..., 4:-nm], axis=-1) > conf_threshold
        x = x[mask]
        x = np.c_[
            x[..., :4],
            np.amax(x[..., 4:-nm], axis=-1),
            np.argmax(x[..., 4:-nm], axis=-1),
            x[..., -nm:],
        ]

        keep = np.array(
            cv2.dnn.NMSBoxes(
                x[:, :4].tolist(), x[:, 4].tolist(), conf_threshold, iou_threshold
            )
        ).flatten()

        results = {}
        if PlatedetOutputType.RAW in return_types:
            results["raw"] = {"boxes": x, "protos": protos}

        if keep.size > 0:
            boxes = x[keep]
            boxes[..., [0, 1]] -= boxes[..., [2, 3]] / 2
            boxes[..., [2, 3]] += boxes[..., [0, 1]]
            boxes[..., :4] -= [pad_w, pad_h, pad_w, pad_h]
            boxes[..., :4] /= min(ratio)
            boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, im0.shape[1])
            boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, im0.shape[0])
            crops = self.extract_crops(im0, boxes)

            if PlatedetOutputType.NP_ARRAY in return_types:
                results["np_array"] = {"crops": crops, "confidences": x[keep, 4]}
            if PlatedetOutputType.PIL in return_types:
                pil_crops = [Image.fromarray(crop) for crop in crops]
                results["pil"] = {"images": pil_crops, "confidences": x[keep, 4]}
            if PlatedetOutputType.BOXES in return_types:
                results["boxes"] = {
                    "boxes": boxes[..., :4].astype(np.int32),
                    "confidences": x[keep, 4],
                }

        return results

    def extract_crops(self, image: np.ndarray, boxes: np.ndarray) -> List[np.ndarray]:
        crops = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            crops.append(image[y1:y2, x1:x2])
        return crops

    def prepare_input(
        self,
        image: np.ndarray,
        conf_threshold: float,
        iou_threshold: float,
        nm: int = 32,
        return_types: List[Union[PlatedetOutputType, str]] = [PlatedetOutputType.PIL],
    ) -> Dict[str, Any]:
        return_types = self._normalize_return_types(return_types)
        return self.extract_roi(
            np.array(image),
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            nm=nm,
            return_types=return_types,
        )

    def inference(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.6,
        iou_threshold: float = 0.4,
        return_types: List[Union[PlatedetOutputType, str]] = [PlatedetOutputType.PIL],
    ) -> Dict[str, Any]:
        return_types = self._normalize_return_types(return_types)
        return self.prepare_input(
            image,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            return_types=return_types,
        )

    def _normalize_return_types(
        self, return_types: List[Union[PlatedetOutputType, str]]
    ) -> List[PlatedetOutputType]:
        normalized_return_types = []
        for rtype in return_types:
            if isinstance(rtype, PlatedetOutputType):
                normalized_return_types.append(rtype)
            elif isinstance(rtype, str):
                try:
                    normalized_return_types.append(PlatedetOutputType(rtype))
                except ValueError:
                    raise ValueError(f"Invalid return type: {rtype}")
        return normalized_return_types
