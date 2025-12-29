import cv2
import numpy as np


class EdgeMapping:
    @staticmethod
    def get_edge_mapping(
        frame, edge_low_threshold=30, edge_high_threshold=90, kernel_size=3
    ):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(image, edge_low_threshold, edge_high_threshold)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

        # 3. Apply Dilation (Thicken the edges)
        # You can increase the 'iterations' parameter here to make the lines even thicker
        dilated_edges = cv2.dilate(edges, kernel, iterations=4)

        blurred_edges = cv2.GaussianBlur(dilated_edges, (3, 3), 0)

        # 5. Normalize
        # Normalize the blurred map, which now has grayscale values for the edges
        return blurred_edges

    @staticmethod
    def modify_depth_map_with_edges(frame, normalized_depth):
        edge_mapping = EdgeMapping.get_edge_mapping(frame)
        normalized_edges = edge_mapping.astype(np.float32) / 255.0
        modified_depth = np.maximum(normalized_depth, normalized_edges)
        return modified_depth
