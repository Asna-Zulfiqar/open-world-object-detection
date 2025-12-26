import torch
import cv2
from groundingdino.util.inference import load_model, predict


class GroundingDINODetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        config_path = "models/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        checkpoint_path = "models/grounding_dino/groundingdino_swint_ogc.pth"

        self.model = load_model(
            model_config_path=config_path,
            model_checkpoint_path=checkpoint_path,
            device=self.device,
        )

    def detect(self, image, prompt="an object", box_threshold=0.3, text_threshold=0.25):
        # Convert BGR to RGB if needed
        if image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Convert to float32 and scale 0-1
        image_tensor = torch.from_numpy(image_rgb.astype('float32') / 255.0)
        # Channels first (C, H, W)
        image_tensor = image_tensor.permute(2, 0, 1)
        # Move to device
        image_tensor = image_tensor.to(self.device)
        
        # Pass tensor directly (not in a list)
        boxes, scores, phrases = predict(
            model=self.model,
            image=image_tensor,
            caption=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        h, w, _ = image.shape
        detections = []
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            detections.append({
                "bbox": [int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)],
                "confidence": float(score),
            })
        return detections