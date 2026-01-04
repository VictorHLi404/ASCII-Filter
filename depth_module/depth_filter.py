import torch
import cv2
import numpy as np
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image


class DepthFilter:

    def __init__(self, model_type: str):

        # --- Configuration ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.depth_processor = AutoImageProcessor.from_pretrained(model_type)
            self.depth_model = AutoModelForDepthEstimation.from_pretrained(
                model_type
            ).to(self.device)
        except Exception as e:
            print(f"Error loading Depth Filter: {e}")
            exit()

    def get_normalized_depth_map(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)

        inputs = self.depth_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.depth_model(**inputs)

        processed_outputs = self.depth_processor.post_process_depth_estimation(
            outputs, target_sizes=[(image.height, image.width)]
        )

        predicted_depth = processed_outputs[0]["predicted_depth"]
        depth_map = predicted_depth.detach().cpu().numpy().squeeze()

        depth_min = depth_map.min()
        depth_max = depth_map.max()
        normalized_depth = (
            (depth_map - depth_min) / (depth_max - depth_min)
            if depth_max - depth_min > 0
            else np.zeros_like(depth_map)
        )

        if depth_max - depth_min > 0:
            normalized_depth_inverted = 1.0 - normalized_depth
        else:
            normalized_depth_inverted = np.zeros_like(depth_map)

        return normalized_depth_inverted
