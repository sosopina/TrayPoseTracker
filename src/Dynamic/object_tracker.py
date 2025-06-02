import cv2
import numpy as np
from pyorbbecsdk import *
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common.utils import frame_to_bgr_image

class ObjectTracker:
    def __init__(self):
        self.trajectory = []
        self.max_trajectory_length = 30  # Maximum number of points to keep in trajectory
        self.min_contour_area = 100  # Increased minimum area to avoid hand detection
        self.smooth_trajectory = []  # For smooth trajectory visualization
        self.smooth_window = 5  # Window size for trajectory smoothing
        
    def detect_object(self, frame):

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([20, 100, 100])  # Lower bound for yellow
        upper_yellow = np.array([30, 255, 255])  # Upper bound for yellow
        
        # Create mask for yellow
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply Gaussian blur to smooth the mask
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Filter out very small contours (noise)
            if cv2.contourArea(largest_contour) < self.min_contour_area:
                return None, None
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Calculate center point
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Draw debug information
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 255), 2)
            
            return (center_x, center_y), (x, y, w, h)
        
        return None, None

    def update_trajectory(self, point):
        """
        Update the trajectory with new point and maintain smooth trajectory
        """
        self.trajectory.append(point)
        if len(self.trajectory) > self.max_trajectory_length:
            self.trajectory.pop(0)
        
        # Update smooth trajectory
        self.smooth_trajectory.append(point)
        if len(self.smooth_trajectory) > self.smooth_window:
            self.smooth_trajectory.pop(0)

    def get_smooth_point(self, points):
        """
        Calculate smoothed point from recent trajectory points
        """
        if not points:
            return None
        x = sum(p[0] for p in points) / len(points)
        y = sum(p[1] for p in points) / len(points)
        return (int(x), int(y))

    def draw_trajectory(self, frame):
        """
        Draw the smooth trajectory on the frame
        """
        if len(self.smooth_trajectory) < 2:
            return
            
        # Draw smooth trajectory
        for i in range(1, len(self.smooth_trajectory)):
            # Calculate smooth points
            p1 = self.get_smooth_point(self.smooth_trajectory[max(0, i-2):i+1])
            p2 = self.get_smooth_point(self.smooth_trajectory[max(0, i-1):i+2])
            
            if p1 and p2:
                # Draw line with decreasing opacity based on age
                alpha = 1.0 - (i / len(self.smooth_trajectory))
                color = (0, int(255 * alpha), 255)  # Yellow color
                cv2.line(frame, p1, p2, color, 2)

def main():
    # Initialize Orbbec camera
    config = Config()
    pipeline = Pipeline()
    
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            # Use larger frame size (720p)
            color_profile = profile_list.get_video_stream_profile(1280, 720, OBFormat.RGB, 30)
        except OBError as e:
            print(e)
            color_profile = profile_list.get_default_video_stream_profile()
            print("color profile: ", color_profile)
        config.enable_stream(color_profile)
    except Exception as e:
        print(e)
        return

    # Initialize object tracker
    tracker = ObjectTracker()
    
    # Start pipeline
    pipeline.start(config)
    
    while True:
        try:
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue
                
            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue
                
            # Convert to BGR format
            frame = frame_to_bgr_image(color_frame)
            if frame is None:
                print("failed to convert frame to image")
                continue
            
            # Apply slight Gaussian blur to reduce noise
            frame = cv2.GaussianBlur(frame, (3, 3), 0)
            
            # Detect object
            center, bbox = tracker.detect_object(frame)
            
            if center is not None:
                # Update trajectory
                tracker.update_trajectory(center)
                
                # Draw bounding box
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Yellow box
                
                # Draw current position
                cv2.circle(frame, center, 5, (0, 255, 255), -1)  # Yellow dot
                
                # Draw trajectory
                tracker.draw_trajectory(frame)
                
                # Add current position text
                cv2.putText(frame, f"Position: ({center[0]}, {center[1]})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Show frame
            cv2.imshow("Yellow Object Tracking", frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except KeyboardInterrupt:
            break
            
    cv2.destroyAllWindows()
    pipeline.stop()

if __name__ == "__main__":
    main() 



