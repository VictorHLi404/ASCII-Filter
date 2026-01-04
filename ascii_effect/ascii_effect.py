import cv2
import numpy as np


class AsciiEffect:

    @staticmethod
    def calculate_ascii_index_and_brightness(
        frame,
        normalized_depth,
        edge_mapping,
        W_DOTS,
        ATTENUATION_STRENGTH,
        GAMMA,
        MIN_BRIGHTNESS_FLOOR,
        PALETTE_THRESHOLD,
        ASCII_SETTINGS,
    ):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray_frame.shape
        H_DOTS = int(h * W_DOTS / w)

        small_gray_frame = cv2.resize(
            gray_frame, (W_DOTS, H_DOTS), interpolation=cv2.INTER_NEAREST
        )
        small_depth_map = cv2.resize(
            normalized_depth, (W_DOTS, H_DOTS), interpolation=cv2.INTER_NEAREST
        )
        small_edge_mapping = cv2.resize(
            edge_mapping, (W_DOTS, H_DOTS), interpolation=cv2.INTER_NEAREST
        )

        # 2. Apply Depth Attenuation: B_eff = B * (1 - D_norm * Strength)
        attenuation_factor = 1.0 - (small_depth_map * ATTENUATION_STRENGTH)
        effective_brightness = small_gray_frame * attenuation_factor
        effective_brightness = np.clip(effective_brightness, 0, 255)

        # 3. APPLY HIGH CONTRAST (GAMMA CORRECTION)
        normalized_brightness = effective_brightness / 255.0
        gamma_corrected_brightness = np.power(normalized_brightness, GAMMA)
        final_brightness = (gamma_corrected_brightness * 255.0).astype(np.float32)
        final_brightness = np.clip(final_brightness, 0, 255)
        final_brightness = np.maximum(final_brightness, small_edge_mapping)

        final_brightness = np.maximum(final_brightness, MIN_BRIGHTNESS_FLOOR)

        # 4. MULTI-PALETTE MAPPING
        final_normalized = final_brightness / 255.0
        ascii_index_grid = np.zeros_like(final_brightness, dtype=np.uint8)

        # --- A. Process the DARK (Lower) Brightness Range ---
        dark_mask = final_normalized <= PALETTE_THRESHOLD
        if np.any(dark_mask):
            dark_values = final_normalized[dark_mask]
            remapped_dark_values = dark_values / PALETTE_THRESHOLD
            num_dark_chars = len(ASCII_SETTINGS.DARK_CHARS)
            dark_indices = (remapped_dark_values * num_dark_chars).astype(np.uint8)
            dark_indices = np.clip(dark_indices, 0, num_dark_chars - 1)
            ascii_index_grid[dark_mask] = dark_indices

        # --- B. Process the BRIGHT (Upper) Brightness Range ---
        bright_mask = final_normalized > PALETTE_THRESHOLD
        if np.any(bright_mask):
            bright_values = final_normalized[bright_mask]
            range_size = 1.0 - PALETTE_THRESHOLD
            remapped_bright_values = (bright_values - PALETTE_THRESHOLD) / range_size
            num_bright_chars = len(ASCII_SETTINGS.BRIGHT_CHARS)
            bright_indices = (remapped_bright_values * num_bright_chars).astype(
                np.uint8
            )
            bright_indices = np.clip(bright_indices, 0, num_bright_chars - 1)
            offset = len(ASCII_SETTINGS.DARK_CHARS)
            ascii_index_grid[bright_mask] = bright_indices + offset

        return ascii_index_grid

    @staticmethod
    def create_ascii_frame_with_density(
        brightness_indices, original_frame, ascii_settings
    ):

        color_breakpoint = len(ascii_settings.DARK_CHARS)
        charset = ascii_settings.DARK_CHARS + ascii_settings.BRIGHT_CHARS
        H_DOTS, W_DOTS = brightness_indices.shape

        target_h = original_frame.shape[0]
        target_w = original_frame.shape[1]

        float_block_w = target_w / W_DOTS
        float_block_h = target_h / H_DOTS

        FONT = cv2.FONT_HERSHEY_SIMPLEX

        FONT_SCALE = (float_block_h / 20.0) * 0.9
        FONT_SCALE = max(0.1, FONT_SCALE)

        THICKNESS = max(1, int(FONT_SCALE * 1.5))

        (text_w, text_h), baseline = cv2.getTextSize("W", FONT, FONT_SCALE, THICKNESS)
        output_frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        # --- Loop through the index grid and draw the corresponding character ---
        for i in range(H_DOTS):
            for j in range(W_DOTS):

                index = brightness_indices[i, j]
                index = np.clip(index, 0, len(charset) - 1)

                if index < color_breakpoint:
                    char_color = ascii_settings.DARK_COLOR
                else:
                    char_color = ascii_settings.BRIGHT_COLOR

                char = charset[index]

                x_start = int(j * float_block_w)
                y_start = int(i * float_block_h)

                current_block_w = int((j + 1) * float_block_w) - x_start
                current_block_h = int((i + 1) * float_block_h) - y_start

                draw_x = x_start + (current_block_w - text_w) // 2
                draw_y = y_start + (current_block_h + text_h) // 2

                # Draw the character
                cv2.putText(
                    output_frame,
                    char,
                    (draw_x, draw_y),
                    FONT,
                    FONT_SCALE,
                    char_color,
                    THICKNESS,
                    cv2.LINE_AA,
                )

        return output_frame
