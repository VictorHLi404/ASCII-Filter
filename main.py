import cv2
import numpy as np
from depth_module.depth_filter import DepthFilter
from ascii_effect.ascii_effect import AsciiEffect
from dataclasses import dataclass
from typing import Optional
import json


@dataclass
class FilterSettings:
    W_DOTS: int
    ATTENUATION_STRENGTH: int
    GAMMA: int
    PALETTE_THRESHOLD: int
    BRIGHTNESS_FLOOR: int


class AsciiVideoEditor:

    SETTINGS_FILE_PATH = "settings/settings.json"
    BRIGHT_CHARS = list("0Zwpbho#W8B")
    DARK_CHARS = list('.`":I!>~_?')
    ASCII_MASTER_CHARS = DARK_CHARS + BRIGHT_CHARS

    def __init__(self):
        if self.load_saved_settings() is not None:
            self.settings = self.load_saved_settings()
        else:
            self.settings = self.load_default_settings()

    def load_default_settings(self) -> FilterSettings:
        try:
            with open(AsciiVideoEditor.SETTINGS_FILE_PATH, "r") as file:
                json_settings = json.load(file)
                default_settings = json_settings["DEFAULT"]
                filter_settings = FilterSettings(
                    W_DOTS=default_settings["W_DOTS"],
                    ATTENUATION_STRENGTH=default_settings["ATTENUATION_STRENGTH"],
                    GAMMA=default_settings["GAMMA"],
                    PALETTE_THRESHOLD=default_settings["PALETTE_THRESHOLD"],
                    BRIGHTNESS_FLOOR=default_settings["BRIGHTNESS_FLOOR"],
                )
                return filter_settings
        except:
            raise Exception("Failed to successfully parse default settings.")

    def load_saved_settings(self) -> Optional[FilterSettings]:
        try:
            with open(AsciiVideoEditor.SETTINGS_FILE_PATH, "r") as file:
                json_settings = json.load(file)
                if "SAVED" not in json_settings:
                    return None
                saved_settings = json_settings["SAVED"]
                filter_settings = FilterSettings(
                    W_DOTS=saved_settings["W_DOTS"],
                    ATTENUATION_STRENGTH=saved_settings["ATTENUATION_STRENGTH"],
                    GAMMA=saved_settings["GAMMA"],
                    PALETTE_THRESHOLD=saved_settings["PALETTE_THRESHOLD"],
                    BRIGHTNESS_FLOOR=saved_settings["BRIGHTNESS_FLOOR"],
                )
                return filter_settings
        except:
            raise Exception("Failed to successfully parse saved settings.")

    def save_settings(self, filter_settings: FilterSettings) -> None:
        try:
            try:
                with open(AsciiVideoEditor.SETTINGS_FILE_PATH, "r") as file:
                    json_settings = json.load(file)
            except:
                raise Exception(
                    "Failed to read existing filter settins while attempting to save."
                )

            json_settings["SAVED"] = filter_settings
            with open(AsciiVideoEditor.SETTINGS_FILE_PATH, "w") as file:
                json.dump(json_settings, file, indent=4)
        except:
            raise Exception("Failed to save filter settings.")

    def run_display(self):
        # Initialize camera input
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
        depth_filter = DepthFilter(model_type="DPT_Large")
        print("Starting MiDaS processing. Initial frames will be slow (CPU mode).")

        MAIN_WINDOW_NAME = "ACII Depth Filter | Press Q to Quit"
        cv2.namedWindow(MAIN_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

        cv2.createTrackbar(
            "RESOLUTION", MAIN_WINDOW_NAME, self.settings.W_DOTS, 300, lambda x: None
        )
        cv2.setTrackbarMin("RESOLUTION", MAIN_WINDOW_NAME, 40)

        cv2.createTrackbar(
            "DEPTH ATTENUATION",
            MAIN_WINDOW_NAME,  # Use the main window name
            self.settings.ATTENUATION_STRENGTH,
            100,
            lambda x: None,
        )

        # Trackbar 3: GAMMA (Contrast)
        # Range: 1 to 300 (scaled to 0.01 to 3.00)
        cv2.createTrackbar(
            "GAMMA",
            MAIN_WINDOW_NAME,  # Use the main window name
            self.settings.GAMMA,
            300,
            lambda x: None,
        )
        cv2.setTrackbarMin("GAMMA", MAIN_WINDOW_NAME, 1)

        cv2.createTrackbar(
            "PALETTE THRESHOLD",
            MAIN_WINDOW_NAME,
            self.settings.PALETTE_THRESHOLD,
            100,
            lambda x: None,
        )
        cv2.setTrackbarMin("PALETTE THRESHOLD", MAIN_WINDOW_NAME, 1)

        cv2.createTrackbar(
            "BRIGHTNESS FLOOR",
            MAIN_WINDOW_NAME,
            self.settings.BRIGHTNESS_FLOOR,
            255,
            lambda x: None,
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip the frame
            frame = cv2.flip(frame, 1)

            W_DOTS = max(40, cv2.getTrackbarPos("RESOLUTION", MAIN_WINDOW_NAME))
            ATTENUATION_STRENGTH = (
                cv2.getTrackbarPos("DEPTH ATTENUATION", MAIN_WINDOW_NAME) / 100.0
            )
            GAMMA = max(0.01, cv2.getTrackbarPos("GAMMA", MAIN_WINDOW_NAME) / 100.0)
            PALETTE_THRESHOLD = max(
                0.01,
                cv2.getTrackbarPos("PALETTE THRESHOLD", MAIN_WINDOW_NAME) / 100.0,
            )
            BRIGHTNESS_FLOOR = max(
                5, cv2.getTrackbarPos("BRIGHTNESS FLOOR", MAIN_WINDOW_NAME)
            )

            normalized_depth = depth_filter.get_normalized_depth_map(frame)

            # --- NEW CODE BLOCK ---
            # 1. Convert Frame and Depth to Dot Mask (Pointillism - unchanged)

            # ascii_index_grid holds WHICH character to use.
            # final_brightness_grid holds the final 0-255 brightness value.
            ascii_index_grid, final_brightness_grid = (
                AsciiEffect.calculate_ascii_index_and_brightness(
                    frame,
                    normalized_depth,
                    W_DOTS,
                    ATTENUATION_STRENGTH,
                    GAMMA,
                    BRIGHTNESS_FLOOR,
                    PALETTE_THRESHOLD,
                    AsciiVideoEditor.DARK_CHARS,
                    AsciiVideoEditor.BRIGHT_CHARS,
                )
            )

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
            ascii_frame = AsciiEffect.create_ascii_frame_with_density(
                ascii_index_grid,
                density_mask,
                frame,
                AsciiVideoEditor.ASCII_MASTER_CHARS,
            )

            # --- FIX: RESIZE FINAL_FRAME TO MATCH INPUT FRAME HEIGHT ---
            target_height = frame.shape[0]

            # Calculate the new width while preserving aspect ratio
            current_width = ascii_frame.shape[1]
            current_height = ascii_frame.shape[0]

            # We must resize to ensure the height is exactly target_height
            target_width = int(current_width * target_height / current_height)

            final_ascii_frame = cv2.resize(
                ascii_frame,
                (target_width, target_height),
                interpolation=cv2.INTER_NEAREST,
            )

            # 3. Display the final result

            top_row = np.hstack((frame, depth_display))
            bottom_row = np.hstack((final_ascii_frame, final_ascii_frame))

            output_display = np.vstack((top_row, bottom_row))
            cv2.imshow(
                "Depth-Filtered Pointillism (GPU) | Press Q to Quit", output_display
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("s"):
                settings_to_save = {
                    "W_DOTS": max(
                        40, cv2.getTrackbarPos("RESOLUTION", MAIN_WINDOW_NAME)
                    ),
                    "ATTENUATION_STRENGTH": cv2.getTrackbarPos(
                        "DEPTH ATTENUATION", MAIN_WINDOW_NAME
                    ),
                    "GAMMA": cv2.getTrackbarPos("GAMMA", MAIN_WINDOW_NAME),
                    "PALETTE_THRESHOLD": cv2.getTrackbarPos(
                        "PALETTE THRESHOLD", MAIN_WINDOW_NAME
                    ),
                    "BRIGHTNESS_FLOOR": cv2.getTrackbarPos(
                        "BRIGHTNESS FLOOR", MAIN_WINDOW_NAME
                    ),
                }
                self.save_settings(settings_to_save)
                print("Saved current settings to file.")

        cap.release()
        cv2.destroyAllWindows()
        print("Depth processing stopped.")


if __name__ == "__main__":
    video_editor = AsciiVideoEditor()
    video_editor.run_display()
