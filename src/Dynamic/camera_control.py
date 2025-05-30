"""
Simple Orbbec Femto Bolt camera control
This module provides basic functionality to access and display the color stream from the Orbbec camera.
"""

import cv2
import os
from datetime import datetime
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, VideoStreamProfile, FrameSet
import sys

# Add the project root to Python path to access common modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common.utils import frame_to_bgr_image

# Define paths for saving data
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset")
SCREENSHOTS_DIR = os.path.join(DATASET_DIR, "screenshots")
RECORDINGS_DIR = os.path.join(DATASET_DIR, "recordings")

# Create directories if they don't exist
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
os.makedirs(RECORDINGS_DIR, exist_ok=True)

# Display settings
DISPLAY_WIDTH = 1080  # Width of the display window
DISPLAY_HEIGHT = 720  # Height of the display window

def initialize_camera():
    """
    Initialize the Orbbec camera pipeline with color stream configuration.
    Returns:
        pipeline: Configured pipeline object
    """
    # Create pipeline and config objects
    pipeline = Pipeline()
    config = Config()
    
    # Get available color stream profiles
    profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    
    # Try to get 1280x720 RGB stream at 30fps, fallback to default if not available
    try:
        color_profile = profile_list.get_video_stream_profile(1280, 0, OBFormat.RGB, 30)
    except:
        color_profile = profile_list.get_default_video_stream_profile()
    
    # Enable the color stream
    config.enable_stream(color_profile)
    
    # Start the pipeline with our configuration
    pipeline.start(config)
    return pipeline

def get_color_frame(pipeline):
    """
    Get a single color frame from the camera.
    Args:
        pipeline: Active pipeline object
    Returns:
        color_image: BGR formatted image or None if failed
    """
    # Wait for frames with 100ms timeout
    frames = pipeline.wait_for_frames(100)
    if frames is None:
        return None
        
    # Extract color frame
    color_frame = frames.get_color_frame()
    if color_frame is None:
        return None
        
    # Convert to BGR format for OpenCV
    return frame_to_bgr_image(color_frame)

def save_screenshot(image):
    """
    Save a screenshot with timestamp.
    Args:
        image: BGR image to save
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(SCREENSHOTS_DIR, f"screenshot_{timestamp}.jpg")
    cv2.imwrite(filename, image)
    print(f"Screenshot saved: {filename}")

def add_text_overlay(image, is_recording):
    """
    Add text overlay to the image showing controls and recording status.
    Args:
        image: BGR image to add text to
        is_recording: Boolean indicating if recording is active
    Returns:
        image: Image with text overlay
    """
    # Create a copy of the image to avoid modifying the original
    display_image = image.copy()
    
    # Define text properties
    font = cv2.FONT_HERSHEY_DUPLEX  # Changed to a more readable font
    font_scale = 0.6  # Increased font size
    font_thickness = 1  # Reduced thickness for cleaner look
    font_color = (255, 255, 255)  # White color for better visibility
    
    # Add semi-transparent background for text
    overlay = display_image.copy()
    cv2.rectangle(overlay, (5, 5), (400, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, display_image, 0.4, 0, display_image)
    
    # Add control instructions
    cv2.putText(display_image, "Press 's' to save image", (15, 35), font, font_scale, font_color, font_thickness)
    cv2.putText(display_image, "Press 'r' to start/stop recording", (15, 65), font, font_scale, font_color, font_thickness)
    cv2.putText(display_image, "Press 'ESC' to quit", (15, 95), font, font_scale, font_color, font_thickness)
    
    # Add recording status if recording
    if is_recording:
        cv2.putText(display_image, "RECORDING", (20, 125), font, font_scale, (0, 0, 255), font_thickness)
    
    return display_image

def main():
    """
    Main function to demonstrate camera usage
    """
    # Initialize camera
    pipeline = initialize_camera()
    
    # Initialize video writer
    video_writer = None
    is_recording = False
    
    try:
        while True:
            # Get color frame
            color_image = get_color_frame(pipeline)
            if color_image is None:
                continue
            
            # Resize the frame for display
            display_image = cv2.resize(color_image, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            
            # Add text overlay
            display_image = add_text_overlay(display_image, is_recording)
                
            # Display the image
            cv2.imshow("Orbbec Camera", display_image)
            
            # Handle key presses
            key = cv2.waitKey(1)
            
            # Take screenshot on 's' key
            if key == ord('s'):
                save_screenshot(color_image)  # Save original resolution image
            
            # Toggle recording on 'r' key
            elif key == ord('r'):
                if not is_recording:
                    # Start recording
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(RECORDINGS_DIR, f"recording_{timestamp}.mp4")
                    height, width = color_image.shape[:2]
                    video_writer = cv2.VideoWriter(
                        filename,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        30.0,
                        (width, height)
                    )
                    is_recording = True
                    print(f"Started recording: {filename}")
                else:
                    # Stop recording
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                    is_recording = False
                    print("Stopped recording")
            
            # Break loop on 'q' or ESC key
            elif key in (ord('q'), 27):  # 27 is ESC key
                break
            
            # Write frame if recording
            if is_recording and video_writer is not None:
                video_writer.write(color_image)  # Save original resolution video
                
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        pipeline.stop()

if __name__ == "__main__":
    main()
