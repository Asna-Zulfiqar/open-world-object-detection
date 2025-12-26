import torch
import numpy as np
from groundingdino.util.inference import load_model, predict

class GroundingDINODetector:
    def __init__(
        self,
        config_path=None,
        checkpoint_path=None,
        device=None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = load_model(
            model_config_path=config_path,
            model_checkpoint_path=checkpoint_path,
            device=self.device
        )

    def detect(
        self,
        image: np.ndarray,
        prompt: str = "an object",
        box_threshold: float = 0.3,
        text_threshold: float = 0.25
    ):
        """
        Returns class-agnostic bounding boxes
        """
        boxes, scores, phrases = predict(
            model=self.model,
            image=image,
            caption=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )

        detections = []
        h, w, _ = image.shape

        for box, score in zip(boxes, scores):
            x1 = int(box[0] * w)
            y1 = int(box[1] * h)
            x2 = int(box[2] * w)
            y2 = int(box[3] * h)

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": float(score)
            })

        return detections
