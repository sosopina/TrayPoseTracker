"""
RULA (Rapid Upper Limb Assessment) Pose Analysis
This module analyzes human poses from the camera feed using RULA methodology.
"""

import cv2
import os
import numpy as np
from datetime import datetime
import mediapipe as mp
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
DISPLAY_WIDTH = 1080
DISPLAY_HEIGHT = 720

# RULA scoring thresholds and feedback messages
RULA_THRESHOLDS = {
    'upper_arm': {
        'warning': 20,  # degrees from vertical
        'critical': 45,  # degrees from vertical
        'message': "Keep your upper arm closer to your body"
    },
    'lower_arm': {
        'warning': 60,  # degrees from horizontal
        'critical': 100,  # degrees from horizontal
        'message': "Keep your forearm more horizontal"
    },
    'wrist': {
        'warning': 15,  # degrees from neutral
        'critical': 30,  # degrees from neutral
        'message': "Keep your wrist in a neutral position"
    },
    'neck': {
        'warning': 10,  # degrees from vertical
        'critical': 20,  # degrees from vertical
        'message': "Keep your neck straight"
    },
    'trunk': {
        'warning': 10,  # degrees from vertical
        'critical': 20,  # degrees from vertical
        'message': "Keep your trunk straight"
    }
}

# Define key body parts for RULA analysis
BODY_PARTS = {
    'nose': mp.solutions.pose.PoseLandmark.NOSE,
    'right_shoulder': mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
    'left_shoulder': mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
    'right_elbow': mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
    'left_elbow': mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
    'right_wrist': mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
    'left_wrist': mp.solutions.pose.PoseLandmark.LEFT_WRIST,
    'right_hip': mp.solutions.pose.PoseLandmark.RIGHT_HIP,
    'left_hip': mp.solutions.pose.PoseLandmark.LEFT_HIP,
    'right_knee': mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
    'left_knee': mp.solutions.pose.PoseLandmark.LEFT_KNEE,
    'right_ankle': mp.solutions.pose.PoseLandmark.RIGHT_ANKLE,
    'left_ankle': mp.solutions.pose.PoseLandmark.LEFT_ANKLE
}

class RULAAnalyzer:
    def __init__(self):
        """Initialize the RULA analyzer with MediaPipe pose detection"""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.warnings = []
        self.last_warning_time = datetime.now()
        self.warning_cooldown = 5  # seconds between warnings

    def get_landmark_position(self, landmarks, body_part):
        """Get the position of a specific body part landmark"""
        if landmarks and body_part in BODY_PARTS:
            landmark = landmarks[BODY_PARTS[body_part]]
            if landmark.visibility > 0.5:  # Only return if the landmark is visible
                return landmark
        return None

    def calculate_angles(self, landmarks):
        """Calculate relevant angles for RULA analysis"""
        angles = {}
        
        # Upper arm angle (relative to vertical)
        right_shoulder = self.get_landmark_position(landmarks, 'right_shoulder')
        right_elbow = self.get_landmark_position(landmarks, 'right_elbow')
        if right_shoulder and right_elbow:
            angles['upper_arm'] = self._calculate_angle(right_shoulder, right_elbow, vertical=True)
        
        # Lower arm angle (relative to upper arm)
        right_wrist = self.get_landmark_position(landmarks, 'right_wrist')
        if right_elbow and right_wrist:
            angles['lower_arm'] = self._calculate_angle(right_elbow, right_wrist, vertical=False)
        
        # Neck angle (relative to vertical)
        nose = self.get_landmark_position(landmarks, 'nose')
        if nose and right_shoulder:
            angles['neck'] = self._calculate_angle(nose, right_shoulder, vertical=True)
        
        # Trunk angle (relative to vertical)
        right_hip = self.get_landmark_position(landmarks, 'right_hip')
        if right_shoulder and right_hip:
            angles['trunk'] = self._calculate_angle(right_shoulder, right_hip, vertical=True)
        
        return angles

    def _calculate_angle(self, point1, point2, vertical=False):
        """Calculate angle between two points"""
        if vertical:
            # Calculate angle relative to vertical
            dx = point2.x - point1.x
            dy = point2.y - point1.y
            return np.degrees(np.arctan2(dx, dy))
        else:
            # Calculate angle between two points
            dx = point2.x - point1.x
            dy = point2.y - point1.y
            return np.degrees(np.arctan2(dy, dx))

    def check_posture(self, angles):
        """Check posture and generate warnings"""
        current_time = datetime.now()
        if (current_time - self.last_warning_time).seconds < self.warning_cooldown:
            return self.warnings

        self.warnings = []
        for body_part, angle in angles.items():
            if body_part in RULA_THRESHOLDS:
                threshold = RULA_THRESHOLDS[body_part]
                abs_angle = abs(angle)
                
                if abs_angle >= threshold['critical']:
                    self.warnings.append(f"CRITICAL: {threshold['message']}")
                    self.last_warning_time = current_time
                elif abs_angle >= threshold['warning']:
                    self.warnings.append(f"Warning: {threshold['message']}")
                    self.last_warning_time = current_time

        return self.warnings

    def draw_landmarks(self, frame, landmarks):
        """Draw landmarks and connections on the frame"""
        if landmarks:
            # Draw all landmarks
            self.mp_drawing.draw_landmarks(
                frame,
                landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Draw key points for RULA analysis with different colors
            for body_part, landmark_idx in BODY_PARTS.items():
                if landmarks.landmark[landmark_idx].visibility > 0.5:
                    landmark = landmarks.landmark[landmark_idx]
                    h, w, _ = frame.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    
                    # Choose color based on body part
                    if 'shoulder' in body_part:
                        color = (0, 255, 0)  # Green for shoulders
                    elif 'elbow' in body_part:
                        color = (255, 0, 0)  # Blue for elbows
                    elif 'wrist' in body_part:
                        color = (0, 0, 255)  # Red for wrists
                    elif 'hip' in body_part:
                        color = (255, 255, 0)  # Cyan for hips
                    else:
                        color = (255, 255, 255)  # White for others
                    
                    cv2.circle(frame, (x, y), 5, color, -1)
                    cv2.putText(frame, body_part.replace('_', ' '), (x + 10, y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def analyze_pose(self, frame):
        """Analyze the pose in the given frame"""
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Draw landmarks and connections
            self.draw_landmarks(frame, results.pose_landmarks)
            
            # Calculate angles
            angles = self.calculate_angles(results.pose_landmarks.landmark)
            
            # Check posture and get warnings
            warnings = self.check_posture(angles)
            
            # Add angle information to the frame
            y_offset = 35
            for body_part, angle in angles.items():
                cv2.putText(frame, f"{body_part.replace('_', ' ').title()}: {angle:.1f}°", 
                          (15, y_offset), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                y_offset += 30
            
            # Add warnings to the frame
            if warnings:
                y_offset = 155
                for warning in warnings:
                    color = (0, 0, 255) if "CRITICAL" in warning else (0, 255, 255)
                    cv2.putText(frame, warning, (15, y_offset), 
                              cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 1)
                    y_offset += 30
        
        return frame

def initialize_camera():
    """Initialize the Orbbec camera pipeline"""
    pipeline = Pipeline()
    config = Config()
    
    profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    try:
        color_profile = profile_list.get_video_stream_profile(1080, 0, OBFormat.RGB, 30)
    except:
        color_profile = profile_list.get_default_video_stream_profile()
    
    config.enable_stream(color_profile)
    pipeline.start(config)
    return pipeline

def main():
    """Main function for RULA pose analysis"""
    # Initialize camera and RULA analyzer
    pipeline = initialize_camera()
    rula_analyzer = RULAAnalyzer()
    
    try:
        while True:
            # Get color frame
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue
                
            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue
                
            # Convert to BGR format
            color_image = frame_to_bgr_image(color_frame)
            if color_image is None:
                continue
            
            # Resize the frame for display
            display_image = cv2.resize(color_image, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            
            # Analyze pose
            display_image = rula_analyzer.analyze_pose(display_image)
            
            # Add instructions
            cv2.putText(display_image, "Press 's' to save image", (15, 95), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display_image, "Press 'ESC' to quit", (15, 125), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            
            # Display the image
            cv2.imshow("RULA Pose Analysis", display_image)
            
            # Handle key presses
            key = cv2.waitKey(1)
            if key == ord('s'):
                # Save screenshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(SCREENSHOTS_DIR, f"rula_analysis_{timestamp}.jpg")
                cv2.imwrite(filename, display_image)
                print(f"Screenshot saved: {filename}")
            elif key in (ord('q'), 27):  # ESC key
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        pipeline.stop()

if __name__ == "__main__":
    main() 