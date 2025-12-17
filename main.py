import cv2
import numpy as np
from depth_module.depth_filter import DepthFilter
from ascii_effect.ascii_effect import AsciiEffect

# --- Configuration ---

# --- Configuration for Pointillism and Depth ---
W_DOTS = 160        # Grid width for dot calculations (Lower = bigger dots)
W_DOTS_MIN = 40
DOT_SIZE_FACTOR = 0.66    # Size of the rendered dot within its block
ATTENUATION_STRENGTH = 0.7 # How much depth affects brightness (0.0 to 1.0)
GAMMA = 0.9  # Controls the strength of the contrast boost (try 1.5 to 3.0)
ASCII_CHARS = list(".'`^,:\";~+=*o#MW@")

BRIGHT_CHARS = list("0Zwpbho#W8B") # ~20 dense characters

# 2. Lighter Characters (Used for DARKER parts of the image)
# These characters are sparse and let the background show through.
DARK_CHARS = list(".`\":I!>~_?") # ~20 sparse/light characters
ASCII_MASTER_CHARS = DARK_CHARS + BRIGHT_CHARS
PALETTE_THRESHOLD = 0.5
MIN_BRIGHTNESS_FLOOR = 5 # Minimum brightness (0-255) to ensure texture is always present

def depth_loop():
    # Initialize camera input
    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    depth_filter = DepthFilter(model_type="DPT_Large")
    
    print("Starting MiDaS processing. Initial frames will be slow (CPU mode).")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame
        frame = cv2.flip(frame, 1)

        normalized_depth = depth_filter.get_normalized_depth_map(frame)

        # --- NEW CODE BLOCK ---
        # 1. Convert Frame and Depth to Dot Mask (Pointillism - unchanged)
        
        # ascii_index_grid holds WHICH character to use.
        # final_brightness_grid holds the final 0-255 brightness value.
        ascii_index_grid, final_brightness_grid = AsciiEffect.calculate_ascii_index_and_brightness(frame, normalized_depth, W_DOTS, ATTENUATION_STRENGTH, GAMMA, MIN_BRIGHTNESS_FLOOR, PALETTE_THRESHOLD, DARK_CHARS, BRIGHT_CHARS)
        
        # 1.2 Calculate Density Mask (Stochastic Sampling)
        
        # Convert final_brightness_grid [0-255] to density probability [0.0-1.0]
        # Low brightness (background) means low density probability.
        density_probability_grid = final_brightness_grid / 255.0
        
        # Apply Stochastic Sampling (similar to the Pointillism mask):
        random_grid = np.random.rand(*density_probability_grid.shape)
        density_mask = (density_probability_grid > random_grid).astype(np.uint8)

        # Convert the normalized depth map [0, 1] back to a visual 8-bit image [0, 255]
        # Since we INVERTED the depth map (0=Closest, 1=Farthest), 
        # the display image will show: Black (0) = Closest, White (255) = Farthest.
        depth_display = (normalized_depth * 255).astype(np.uint8)
        
        # Convert the grayscale depth map to a 3-channel BGR image for stacking
        depth_display = cv2.cvtColor(depth_display, cv2.COLOR_GRAY2BGR)

        # --- NEW CODE BLOCK ---
        # 1. Convert Frame and Depth to Dot Mask
        
        # 2. Render Dot Mask to the Final Frame
        ascii_frame = AsciiEffect.create_ascii_frame_with_density(ascii_index_grid, density_mask, frame, ASCII_MASTER_CHARS)
        
        # --- FIX: RESIZE FINAL_FRAME TO MATCH INPUT FRAME HEIGHT ---
        target_height = frame.shape[0]
        
        # Calculate the new width while preserving aspect ratio
        current_width = ascii_frame.shape[1]
        current_height = ascii_frame.shape[0]
        
        # We must resize to ensure the height is exactly target_height
        target_width = int(current_width * target_height / current_height)

        final_ascii_frame = cv2.resize(ascii_frame, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        
        # 3. Display the final result

        top_row = np.hstack((frame, depth_display))
        bottom_row = np.hstack((final_ascii_frame, final_ascii_frame))

        output_display = np.vstack((top_row, bottom_row))
        cv2.imshow('Depth-Filtered Pointillism (GPU) | Press Q to Quit', output_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Depth processing stopped.")

if __name__ == '__main__':
    depth_loop()