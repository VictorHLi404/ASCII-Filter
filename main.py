import cv2
import numpy as np
from depth_module.depth_filter import DepthFilter
from ascii_effect.ascii_effect import AsciiEffect
from edge_module.edge_mapping import EdgeMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json

@dataclass
class FilterSettings:
    W_DOTS: int
    ATTENUATION_STRENGTH: int
    GAMMA: int
    PALETTE_THRESHOLD: int
    BRIGHTNESS_FLOOR: int


@dataclass
class AsciiSettings:
    DARK_CHARS: list[str]
    BRIGHT_CHARS: list[str]
    DARK_COLOR: list[int]
    BRIGHT_COLOR: list[int]

class AsciiVideoEditor:

    SETTINGS_FILE_PATH = "settings/settings.json"
    INPUT_VIDEOS_FILE_PATH = "input_videos"
    OUTPUT_VIDEOS_FILE_PATH = "output_videos"

    MAX_DISPLAY_HEIGHT = 320
    MAIN_WINDOW_NAME = "ASCII Depth Filter | Press Q to Quit"
    MODEL_TYPE = "depth-anything/Depth-Anything-V2-Small-hf"

    def __init__(self):
        self.initialize_settings()

    def initialize_settings(self):
        if self.load_saved_settings_from_file() is not None:
            self.adjustable_settings = self.load_saved_settings_from_file()
        else:
            self.adjustable_settings = self.load_default_settings()
        self.fixed_settings = self.load_fixed_settings()
        self.depth_filter = DepthFilter(AsciiVideoEditor.MODEL_TYPE)
        self.first_frame_processed = False

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
    
    def load_fixed_settings(self) -> AsciiSettings:
        try:
            with open(AsciiVideoEditor.SETTINGS_FILE_PATH, "r") as file:
                json_settings = json.load(file)
                fixed_settings = json_settings["FIXED"]
                ascii_settings = AsciiSettings(
                    DARK_CHARS=fixed_settings["DARK_CHARS"],
                    BRIGHT_CHARS=fixed_settings["BRIGHT_CHARS"],
                    DARK_COLOR=fixed_settings["DARK_COLOR"],
                    BRIGHT_COLOR=fixed_settings["BRIGHT_COLOR"]
                )
                return ascii_settings
        except:
            raise Exception("Failed to sucessfully parse fixed settings.")

    def load_saved_settings_from_file(self) -> Optional[FilterSettings]:
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

    def write_settings_to_file(self, filter_settings: FilterSettings) -> None:
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

    def save_settings(self):
        settings_to_save = {
            "W_DOTS": max(
                40,
                cv2.getTrackbarPos("RESOLUTION", AsciiVideoEditor.MAIN_WINDOW_NAME),
            ),
            "ATTENUATION_STRENGTH": cv2.getTrackbarPos(
                "DEPTH ATTENUATION", AsciiVideoEditor.MAIN_WINDOW_NAME
            ),
            "GAMMA": cv2.getTrackbarPos("GAMMA", AsciiVideoEditor.MAIN_WINDOW_NAME),
            "PALETTE_THRESHOLD": cv2.getTrackbarPos(
                "PALETTE THRESHOLD", AsciiVideoEditor.MAIN_WINDOW_NAME
            ),
            "BRIGHTNESS_FLOOR": cv2.getTrackbarPos(
                "BRIGHTNESS FLOOR", AsciiVideoEditor.MAIN_WINDOW_NAME
            ),
        }
        self.write_settings_to_file(settings_to_save)
        print("Saved current settings to file.")

    def setup_trackbars(self):
        cv2.namedWindow(AsciiVideoEditor.MAIN_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

        cv2.createTrackbar(
            "RESOLUTION",
            AsciiVideoEditor.MAIN_WINDOW_NAME,
            self.adjustable_settings.W_DOTS,
            300,
            lambda x: None,
        )
        cv2.setTrackbarMin("RESOLUTION", AsciiVideoEditor.MAIN_WINDOW_NAME, 40)

        cv2.createTrackbar(
            "DEPTH ATTENUATION",
            AsciiVideoEditor.MAIN_WINDOW_NAME,  # Use the main window name
            self.adjustable_settings.ATTENUATION_STRENGTH,
            100,
            lambda x: None,
        )

        # Trackbar 3: GAMMA (Contrast)
        # Range: 1 to 300 (scaled to 0.01 to 3.00)
        cv2.createTrackbar(
            "GAMMA",
            AsciiVideoEditor.MAIN_WINDOW_NAME,  # Use the main window name
            self.adjustable_settings.GAMMA,
            300,
            lambda x: None,
        )
        cv2.setTrackbarMin("GAMMA", AsciiVideoEditor.MAIN_WINDOW_NAME, 1)

        cv2.createTrackbar(
            "PALETTE THRESHOLD",
            AsciiVideoEditor.MAIN_WINDOW_NAME,
            self.adjustable_settings.PALETTE_THRESHOLD,
            100,
            lambda x: None,
        )
        cv2.setTrackbarMin("PALETTE THRESHOLD", AsciiVideoEditor.MAIN_WINDOW_NAME, 1)

        cv2.createTrackbar(
            "BRIGHTNESS FLOOR",
            AsciiVideoEditor.MAIN_WINDOW_NAME,
            self.adjustable_settings.BRIGHTNESS_FLOOR,
            255,
            lambda x: None,
        )

    def process_depth_frame(self, normalized_depth):
        # Convert the normalized depth map [0, 1] back to a visual 8-bit image [0, 255]
        # Since we INVERTED the depth map (0=Closest, 1=Farthest),
        # the display image will show: Black (0) = Closest, White (255) = Farthest.
        depth_frame = (normalized_depth * 255).astype(np.uint8)

        # Convert the grayscale depth map to a 3-channel BGR image for stacking
        depth_display = cv2.cvtColor(depth_frame, cv2.COLOR_GRAY2BGR)
        return depth_display

    def process_ascii_frame(self, frame, normalized_depth):
        ascii_index_grid, final_brightness_grid = (
            AsciiEffect.calculate_ascii_index_and_brightness(
                frame,
                normalized_depth,
                max(
                    40,
                    cv2.getTrackbarPos("RESOLUTION", AsciiVideoEditor.MAIN_WINDOW_NAME),
                ),
                cv2.getTrackbarPos(
                    "DEPTH ATTENUATION", AsciiVideoEditor.MAIN_WINDOW_NAME
                )
                / 100.0,
                max(
                    0.01,
                    cv2.getTrackbarPos("GAMMA", AsciiVideoEditor.MAIN_WINDOW_NAME)
                    / 100.0,
                ),
                max(
                    5,
                    cv2.getTrackbarPos(
                        "BRIGHTNESS FLOOR", AsciiVideoEditor.MAIN_WINDOW_NAME
                    ),
                ),
                max(
                    0.01,
                    cv2.getTrackbarPos(
                        "PALETTE THRESHOLD", AsciiVideoEditor.MAIN_WINDOW_NAME
                    )
                    / 100.0,
                ),
                self.fixed_settings
            )
        )
        # 1.2 Calculate Density Mask (Stochastic Sampling)

        # Convert final_brightness_grid [0-255] to density probability [0.0-1.0]
        # Low brightness (background) means low density probability.
        density_probability_grid = final_brightness_grid / 255.0

        # Apply Stochastic Sampling (similar to the Pointillism mask):
        random_grid = np.random.rand(*density_probability_grid.shape)
        density_mask = (density_probability_grid > random_grid).astype(np.uint8)

        ascii_frame = AsciiEffect.create_ascii_frame_with_density(
            ascii_index_grid,
            density_mask,
            frame,
            self.fixed_settings
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
        return final_ascii_frame

    def create_display_output(self, original_frame, depth_frame, ascii_frame):
        h_orig, w_orig = original_frame.shape[:2]

        scale_factor = AsciiVideoEditor.MAX_DISPLAY_HEIGHT / h_orig
        target_height = AsciiVideoEditor.MAX_DISPLAY_HEIGHT
        target_width = int(w_orig * scale_factor)

        # 1. Resize the Original Frame
        display_original = cv2.resize(original_frame, (target_width, target_height))

        # 2. Resize the Depth Map
        display_depth = cv2.resize(depth_frame, (target_width, target_height))

        # 3. Resize the ASCII Frame
        # To maintain consistency, we force the ASCII frame to match the calculated target size.
        display_ascii = cv2.resize(
            ascii_frame, (target_width, target_height), interpolation=cv2.INTER_NEAREST
        )

        # 4. Stack the frames in the 2x2 layout: (Original, Depth) over (ASCII, ASCII)
        top_row = np.hstack((display_original, display_depth))
        bottom_row = np.hstack((display_ascii, display_ascii))

        output_display = np.vstack((top_row, bottom_row))
        return output_display

    def run_live_camera_display(self):
        # Initialize camera input
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        self.setup_trackbars()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip the frame
            frame = cv2.flip(frame, 1)
            normalized_depth = self.depth_filter.get_normalized_depth_map(frame)
            normalized_depth = EdgeMapping.modify_depth_map_with_edges(frame, normalized_depth)
            depth_frame = self.process_depth_frame(normalized_depth)
            ascii_frame = self.process_ascii_frame(frame, normalized_depth)

            output_display = self.create_display_output(frame, depth_frame, ascii_frame)

            if not self.first_frame_processed:
                display_height, display_width = output_display.shape[:2]
                cv2.resizeWindow(
                    AsciiVideoEditor.MAIN_WINDOW_NAME,
                    display_width,
                    display_height + 80,  # Add space for trackbars
                )
                self.first_frame_processed = True

            cv2.imshow(AsciiVideoEditor.MAIN_WINDOW_NAME, output_display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("s"):
                self.save_settings()

        cap.release()
        cv2.destroyAllWindows()

    def run_live_preview(self, file_path: str):
        full_file_path = AsciiVideoEditor.INPUT_VIDEOS_FILE_PATH + "/" + file_path
        cap = cv2.VideoCapture(full_file_path)
        if not cap.isOpened():
            raise Exception(f"Could not open video file: {file_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        wait_ms = int(1000 / fps) if fps > 0 else 1
        self.setup_trackbars()

        is_paused = False
        ret, current_frame = cap.read()  # Get first frame

        if not ret:
            print("Error: Input video is empty.")
            cap.release()
            return False

        while cap.isOpened():
            key = cv2.waitKey(1 if is_paused else wait_ms) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("s"):
                self.save_settings()
            elif key == ord("p"):
                is_paused = not is_paused
            elif key == ord("e"):
                print("Exporting video...")
                self.batch_process_video(file_path)
                print("Exporting video done! Killing the process now.")
                break

            if not is_paused:
                ret, next_frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret_new, next_frame = cap.read()

            current_frame = next_frame
            normalized_depth = self.depth_filter.get_normalized_depth_map(current_frame)
            normalized_depth = EdgeMapping.modify_depth_map_with_edges(current_frame, normalized_depth)
            depth_frame = self.process_depth_frame(normalized_depth)
            ascii_frame = self.process_ascii_frame(current_frame, normalized_depth)

            output_display = self.create_display_output(
                current_frame, depth_frame, ascii_frame
            )

            if not self.first_frame_processed:
                display_height, display_width = output_display.shape[:2]
                cv2.resizeWindow(
                    AsciiVideoEditor.MAIN_WINDOW_NAME,
                    display_width,
                    display_height + 80,  # Add space for trackbars
                )
                self.first_frame_processed = True

            cv2.imshow(AsciiVideoEditor.MAIN_WINDOW_NAME, output_display)

        cap.release()
        cv2.destroyAllWindows()

    def batch_process_video(self, file_path: str) -> bool:

        input_video_path = Path(AsciiVideoEditor.INPUT_VIDEOS_FILE_PATH) / file_path

        # 2. Generate Output Path: Ensure the output filename is unique and includes the directory.
        # a) Remove the extension from the input filename (e.g., "clip.mp4" -> "clip")
        base_name = Path(file_path).stem
        # b) Construct the final output path string
        output_file_name = f"{base_name}_edited.mp4"
        output_video_path = (
            Path(AsciiVideoEditor.OUTPUT_VIDEOS_FILE_PATH) / output_file_name
        )

        # CRITICAL FIX 1: Ensure the output directory exists
        output_video_path.parent.mkdir(parents=True, exist_ok=True)

        # 3. Setup Video Reader (Capture)
        # Convert Path object to string for cv2.VideoCapture compatibility
        cap = cv2.VideoCapture(str(input_video_path))

        if not cap.isOpened():
            raise Exception(f"Could not open video file: {file_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_size = (input_width, input_height)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_path_str = str(output_video_path.resolve())
        out = cv2.VideoWriter(str(output_path_str), fourcc, fps, frame_size)

        if not out.isOpened():
            print(
                f"FATAL: Could not open VideoWriter. Check codec support on your system."
            )
            cap.release()
            return False

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            normalized_depth = self.depth_filter.get_normalized_depth_map(frame)
            normalized_depth = EdgeMapping.modify_depth_map_with_edges(frame, normalized_depth)
            final_ascii_frame = self.process_ascii_frame(frame, normalized_depth)

            # Case A: Padding (if final_ascii_frame is too narrow)
            if final_ascii_frame.shape[1] < input_width:
                padding_width = input_width - final_ascii_frame.shape[1]
                padding = np.zeros((input_height, padding_width, 3), dtype=np.uint8)
                final_ascii_frame = np.hstack((final_ascii_frame, padding))

            # Case B: Cropping (if final_ascii_frame is too wide—less common)
            elif final_ascii_frame.shape[1] > input_width:
                final_ascii_frame = final_ascii_frame[:, :input_width]

            # 5. Write Frame
            out.write(final_ascii_frame)

            frame_count += 1
            if frame_count % 300 == 0:
                progress = (frame_count / total_frames) * 100
                print(
                    f"   Processed {frame_count} / {total_frames} frames ({progress:.1f}%)"
                )

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        return True

    def assert_video_exists(self, file_path: str) -> bool:
        return Path(AsciiVideoEditor.INPUT_VIDEOS_FILE_PATH + "/" + file_path).exists()


if __name__ == "__main__":
    """
    TODO: implement sobel detection for better clarity, add new parameter to influence how much it factors into the final image
    TODO: MAYBE do the acerola thing where edges are changed depending on angle? 
    """
    video_editor = AsciiVideoEditor()
    mode = input(
        "Enter 1 to enter live camera display mode, or Enter 2 to enter video editing mode: "
    )
    while mode not in ["1", "2"]:
        mode = input(
            "Enter 1 to enter live camera display mode, or Enter 2 to enter video editing mode: "
        )
    if mode == "1":
        video_editor.run_live_camera_display()
    else:
        while True:
            file_path = input(
                "Copy in the file name for the video you want to edit. Ensure that the video is placed inside of the input_videos folder: "
            )
            while video_editor.assert_video_exists(file_path) is False:
                file_path = input(
                    "Failed to load file correctly. Please enter the file name again, or recheck the folder: "
                )
            video_editor.run_live_preview(file_path)
            video_editor.initialize_settings()
            answer = input("Do you want to edit another video? Y/N: ")
            if "N" in answer:
                break
