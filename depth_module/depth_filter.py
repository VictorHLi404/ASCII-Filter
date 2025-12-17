import torch
import cv2
import numpy as np

class DepthFilter:

    def __init__(self, model_type: str):
        # --- Configuration ---
        # Use the CPU for now since the GPU installation failed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading MiDaS model on device: {self.device}...")

        # Load MiDaS model (MiDaS_small is fast)
        self.model_type = model_type
        try:
            self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type)
            self.midas.to(self.device)
            self.midas.eval()
            
            # Load MiDaS transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

            if self.model_type == "DPT_Large" or self.model_type == "DPT_Hybrid":
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform
            
            print(f"MiDaS model '{self.model_type}' loaded successfully.")

        except Exception as e:
            print(f"Error loading MiDaS model or dependencies: {e}")
            print("Please ensure you installed torch, torchvision, torchaudio, and timm.")
            exit()

    def get_normalized_depth_map(self, frame):
        """Generates and normalizes the depth map using MiDaS."""
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply MiDaS transformation (resizing, normalization)
        input_batch = self.transform(img).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)
            
            # Resize the output prediction to the original frame size
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        
        # Normalize depth map to range [0, 1] for visual/functional use
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        normalized_depth = (depth_map - depth_min) / (depth_max - depth_min) if depth_max - depth_min > 0 else np.zeros_like(depth_map)

        if depth_max - depth_min > 0:
            normalized_depth_inverted = 1.0 - normalized_depth
        else:
            normalized_depth_inverted = np.zeros_like(depth_map)

        return normalized_depth_inverted