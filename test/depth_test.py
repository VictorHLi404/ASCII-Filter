import torch
import cv2
import numpy as np

# --- Configuration ---
# Use the CPU for now since the GPU installation failed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading MiDaS model on device: {device}...")

# Load MiDaS model (MiDaS_small is fast)
model_type = "DPT_Large"

# --- Configuration for Pointillism and Depth ---
W_DOTS = 120          # Grid width for dot calculations (Lower = bigger dots)
DOT_SIZE_FACTOR = 0.66    # Size of the rendered dot within its block
ATTENUATION_STRENGTH = 0.66 # How much depth affects brightness (0.0 to 1.0)
GAMMA = 3  # Controls the strength of the contrast boost (try 1.5 to 3.0)
ASCII_CHARS = list(" .'`^,:\";~+=*o#MW@")

try:
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device)
    midas.eval()
    
    # Load MiDaS transforms
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
    
    print(f"MiDaS model '{model_type}' loaded successfully.")

except Exception as e:
    print(f"Error loading MiDaS model or dependencies: {e}")
    print("Please ensure you installed torch, torchvision, torchaudio, and timm.")
    exit()

def get_normalized_depth_map(frame):
    """Generates and normalizes the depth map using MiDaS."""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Apply MiDaS transformation (resizing, normalization)
    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        
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

def frame_to_point_grid_ascii(frame, normalized_depth):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray_frame.shape
    H_DOTS = int(h * W_DOTS / w) 
    
    small_gray_frame = cv2.resize(gray_frame, (W_DOTS, H_DOTS), interpolation=cv2.INTER_LINEAR)
    small_depth_map = cv2.resize(normalized_depth, (W_DOTS, H_DOTS), interpolation=cv2.INTER_LINEAR)
    
    # 2. Apply Depth Attenuation: B_eff = B * (1 - D_norm * Strength)
    attenuation_factor = 1.0 - (small_depth_map * ATTENUATION_STRENGTH)
    effective_brightness = small_gray_frame * attenuation_factor
    effective_brightness = np.clip(effective_brightness, 0, 255)


    # --- NEW STEP 3: APPLY HIGH CONTRAST (GAMMA CORRECTION) ---

    
    # Normalize brightness to [0, 1]
    normalized_for_gamma = effective_brightness / 255.0
    
    # Apply the power curve: pushes dark values much darker
    gamma_corrected_brightness = np.power(normalized_for_gamma, GAMMA)
    
    # Scale back to [0, 255]
    final_brightness = (gamma_corrected_brightness * 255.0).astype(np.float32)
    final_brightness = np.clip(final_brightness, 0, 255)

    # Scale the brightness (0-255) to an index within the length of the ASCII_CHARS array
    num_chars = len(ASCII_CHARS)
    
    # We want a high effective_brightness (255) to map to a high index (dense chars like '@')
    # Use integer division to get the index.
    brightness_indices = (final_brightness * num_chars / 256).astype(np.uint8)
    
    # We return the grid of indices, not a dot mask
    return brightness_indices
    

def frame_to_point_grid(frame, normalized_depth):
    # 1. Grayscale and Shrink
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray_frame.shape
    H_DOTS = int(h * W_DOTS / w) 
    
    small_gray_frame = cv2.resize(gray_frame, (W_DOTS, H_DOTS), interpolation=cv2.INTER_LINEAR)
    small_depth_map = cv2.resize(normalized_depth, (W_DOTS, H_DOTS), interpolation=cv2.INTER_LINEAR)
    
    # 2. Apply Depth Attenuation: B_eff = B * (1 - D_norm * Strength)
    attenuation_factor = 1.0 - (small_depth_map * ATTENUATION_STRENGTH)
    effective_brightness = small_gray_frame * attenuation_factor
    effective_brightness = np.clip(effective_brightness, 0, 255)
    
    # 3. Convert brightness [0-255] to probability [0.0-1.0]
    probability_grid = effective_brightness / 255.0
    
    # 4. Apply Stochastic Sampling (Dithering)
    random_grid = np.random.rand(H_DOTS, W_DOTS)
    dot_mask = (probability_grid > random_grid).astype(np.uint8)
    
    return dot_mask

def create_pointillism_frame(dot_mask, original_frame):
    h_mask, w_mask = dot_mask.shape
    target_h = original_frame.shape[0] 
    target_w = int(target_h * w_mask / h_mask)
    
    # If W_DOTS is high, target_h // h_mask can be 0. We must ensure it's at least 1.
    block_h = max(1, target_h // h_mask)
    block_w = max(1, target_w // w_mask)
    out_h = h_mask * block_h
    out_w = w_mask * block_w
    
    output_frame = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    
    for i in range(h_mask):
        for j in range(w_mask):
            if dot_mask[i, j] == 1:
                x_start = j * block_w
                y_start = i * block_h
                
                # # Center the dot in the block
                # dot_w = max(1, int(block_w * DOT_SIZE_FACTOR)) // 2
                # dot_h = max(1, int(block_h * DOT_SIZE_FACTOR)) // 2
                
                # center_x = x_start + block_w // 2
                # center_y = y_start + block_h // 2
                
                # # Draw the white dot
                # output_frame[
                #     center_y - dot_h // 2 : center_y + dot_h // 2,
                #     center_x - dot_w // 2 : center_x + dot_w // 2
                # ] = [255, 255, 255]

                # Calculate the actual dot size (always at least 1)
                dot_w_size = max(1, int(block_w * DOT_SIZE_FACTOR))
                dot_h_size = max(1, int(block_h * DOT_SIZE_FACTOR))

                center_x = x_start + block_w // 2
                center_y = y_start + block_h // 2

                # Draw the white dot using the calculated size
                output_frame[
                    center_y - dot_h_size // 2 : center_y + (dot_h_size + 1) // 2, # +1 for correct rounding/centering
                    center_x - dot_w_size // 2 : center_x + (dot_w_size + 1) // 2
                ] = [255, 255, 255]
                
    return output_frame


def create_ascii_frame(brightness_indices, original_frame):
    H_DOTS, W_DOTS_actual = brightness_indices.shape
    
    target_h = original_frame.shape[0] 
    target_w = original_frame.shape[1] 

    # --- RECALCULATE BLOCK SIZES BASED ON TARGET FRAME ---
    # This calculation is now guaranteed to result in a block size that is at least 1.
    block_w = target_w // W_DOTS_actual
    block_h = target_h // H_DOTS
    
    block_w = max(1, block_w)
    block_h = max(1, block_h)
    
    # --- DYNAMIC FONT SCALE CALCULATION ---
    # FONT_SCALE should be proportional to the block height. 
    # Use 0.8 to give a little space between characters.
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = block_h / 20.0  # Heuristic: 20 is approx. character height at scale 1.0
    FONT_SCALE = FONT_SCALE * 0.9 # Fine-tuning for space
    FONT_SCALE = max(0.1, FONT_SCALE)
    THICKNESS = max(1, int(FONT_SCALE * 1.5)) # Use slightly thicker line for better visibility
    
    # The output frame must match the full size of the original frame for proper stacking
    output_frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # --- Loop through the index grid and draw the corresponding character ---
    for i in range(H_DOTS):
        for j in range(W_DOTS_actual):
            # 1. Get the ASCII character
            index = brightness_indices[i, j]
            # Handle potential overflow if index is out of bounds due to rounding
            index = np.clip(index, 0, len(ASCII_CHARS) - 1)
            char = ASCII_CHARS[index]
            
            # 2. Calculate drawing position
            x = j * block_w
            y = i * block_h
            
            # Position: The Y-coordinate needs to be the text baseline. 
            # We add the block height to the block top (y) to place the baseline.
            cv2.putText(
                output_frame,
                char,
                (x, y + block_h - 1), # Place baseline slightly above the bottom of the block
                FONT,
                FONT_SCALE,
                (255, 255, 255), 
                THICKNESS,
                cv2.LINE_AA
            )
            
    return output_frame


def depth_loop():
    # Initialize camera input
    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Starting MiDaS processing. Initial frames will be slow (CPU mode).")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame
        frame = cv2.flip(frame, 1)

        # # Calculate Depth Map
        # normalized_depth = get_normalized_depth_map(frame)
        
        # # Convert depth map back to a visual 8-bit image for display
        # # The display image shows depth (0=black=closest, 255=white=farthest)
        # depth_display = (normalized_depth * 255).astype(np.uint8)
        # depth_display = cv2.cvtColor(depth_display, cv2.COLOR_GRAY2BGR)


        # Concatenate the original frame and the depth map for comparison


        # cv2.imshow('Original | MiDaS Depth (Press Q to Quit)', output_display)

        # Calculate Depth Map
        normalized_depth = get_normalized_depth_map(frame)
    

    # Convert the normalized depth map [0, 1] back to a visual 8-bit image [0, 255]
        # Since we INVERTED the depth map (0=Closest, 1=Farthest), 
        # the display image will show: Black (0) = Closest, White (255) = Farthest.
        depth_display = (normalized_depth * 255).astype(np.uint8)
        
        # Convert the grayscale depth map to a 3-channel BGR image for stacking
        depth_display = cv2.cvtColor(depth_display, cv2.COLOR_GRAY2BGR)

        # --- NEW CODE BLOCK ---
        # 1. Convert Frame and Depth to Dot Mask
        dot_mask = frame_to_point_grid(frame, normalized_depth)
        ascii_index_grid = frame_to_point_grid_ascii(frame, normalized_depth)
        
        # 2. Render Dot Mask to the Final Frame
        pointillism_frame = create_pointillism_frame(dot_mask, frame)
        ascii_frame = create_ascii_frame(ascii_index_grid, frame)
        
        # --- FIX: RESIZE FINAL_FRAME TO MATCH INPUT FRAME HEIGHT ---
        target_height = frame.shape[0]
        
        # Calculate the new width while preserving aspect ratio
        current_width = pointillism_frame.shape[1]
        current_height = pointillism_frame.shape[0]
        
        # We must resize to ensure the height is exactly target_height
        target_width = int(current_width * target_height / current_height)
        
        final_frame = cv2.resize(
            pointillism_frame, 
            (target_width, target_height), 
            interpolation=cv2.INTER_NEAREST # Use NEAREST to preserve sharp dots
        )

        final_ascii_frame = cv2.resize(ascii_frame, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        
        # 3. Display the final result

        top_row = np.hstack((frame, depth_display))
        bottom_row = np.hstack((final_frame, final_ascii_frame))

        output_display = np.vstack((top_row, bottom_row))
        cv2.imshow('Depth-Filtered Pointillism (GPU) | Press Q to Quit', output_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Depth processing stopped.")

if __name__ == '__main__':
    depth_loop()