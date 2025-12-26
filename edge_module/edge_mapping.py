import cv2
import numpy as np

class EdgeMapping:
    @staticmethod
    def get_edge_mapping(frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(image, 50, 100)
        return edges
    
    @staticmethod
    def modify_depth_map_with_edges(frame, normalized_depth):
        edge_mapping = EdgeMapping.get_edge_mapping(frame)
        inverse_mapping = 1.0 - edge_mapping
        return normalized_depth * inverse_mapping