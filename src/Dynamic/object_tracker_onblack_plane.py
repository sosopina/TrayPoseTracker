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
        self.roi = None # Add attribute to store ROI

    def set_roi(self, x, y, w, h):
        """
        Set the region of interest for object tracking.
        Args:
            x, y: Top-left coordinates of the ROI.
            w, h: Width and height of the ROI.
        """
        self.roi = (x, y, w, h)

    def detect_object(self, frame):

        # If ROI is set, crop the frame to the ROI
        if self.roi:
            x_roi, y_roi, w_roi, h_roi = self.roi
            frame_roi = frame[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
        else:
            # If no ROI is set, return None immediately as we require an ROI for this tracker
            return None, None

        # Convert ROI frame to HSV
        hsv = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2HSV)

        # Define range for yellow color in HSV
        lower_yellow = np.array([20, 100, 100])  # Lower bound for yellow
        upper_yellow = np.array([30, 255, 255])  # Upper bound for yellow

        # Create mask for yellow within the ROI
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # Remove small objects outside the yellow region
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Close small holes inside the yellow region

        # Apply Gaussian blur to smooth the mask
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter for significant contours (you might still want some filtering here based on area or shape if needed)
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.min_contour_area] # Use min_contour_area for basic filtering

        if valid_contours:
            # Get the largest valid contour
            largest_contour = max(valid_contours, key=cv2.contourArea)

            # Get bounding box relative to the ROI
            x_roi_obj, y_roi_obj, w_obj, h_obj = cv2.boundingRect(largest_contour)

            # Calculate center point relative to the ROI
            center_x_roi = x_roi_obj + w_obj // 2
            center_y_roi = y_roi_obj + h_obj // 2

            # Adjust coordinates to get position in original frame
            center_x_orig = center_x_roi + x_roi
            center_y_orig = center_y_roi + y_roi
            x_orig = x_roi_obj + x_roi
            y_orig = y_roi_obj + y_roi

            # The bounding box returned should be in the original frame coordinates
            bbox_orig = (x_orig, y_orig, w_obj, h_obj)

            # Draw debug information on the ORIGINAL frame (optional, depends on which window you want to show it)
            # cv2.drawContours(frame, [largest_contour], -1, (0, 255, 255), 2) # This would draw on the full frame

            # To draw on the ROI window, you would draw on frame_roi before this return
            # cv2.rectangle(frame_roi, (x_roi_obj, y_roi_obj), (x_roi_obj + w_obj, y_roi_obj + h_obj), (0, 255, 255), 2)

            return (center_x_orig, center_y_orig), bbox_orig # Return coordinates in original frame

        return None, None # No object detected in the ROI

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

    def draw_trajectory_on_roi(self, roi_frame, roi_offset_x, roi_offset_y):
        """
        Draw the smooth trajectory on the ROI frame, relative to the ROI's top-left corner.
        """
        if len(self.smooth_trajectory) < 2:
            return

        # Draw smooth trajectory on ROI frame
        for i in range(1, len(self.smooth_trajectory)):
            # Calculate smooth points
            p1_orig = self.get_smooth_point(self.smooth_trajectory[max(0, i-2):i+1])
            p2_orig = self.get_smooth_point(self.smooth_trajectory[max(0, i-1):i+2])

            if p1_orig and p2_orig:
                # Adjust points to be relative to the ROI frame
                p1_roi = (p1_orig[0] - roi_offset_x, p1_orig[1] - roi_offset_y)
                p2_roi = (p2_orig[0] - roi_offset_x, p2_orig[1] - roi_offset_y)

                # Draw line with decreasing opacity based on age
                alpha = 1.0 - (i / len(self.smooth_trajectory))
                color = (0, int(255 * alpha), 255)  # Yellow color
                # Ensure points are within the ROI frame boundaries before drawing (optional but good practice)
                # You would need to check if p1_roi and p2_roi are within (0, 0) to (w_roi, h_roi)
                cv2.line(roi_frame, p1_roi, p2_roi, color, 2)

# Add these variables outside the main loop, perhaps as class attributes or global variables
roi_points = []
selecting_roi = False
roi_selected = False

def select_roi_callback(event, x, y, flags, param):
    global roi_points, selecting_roi, roi_selected, frame # You might need to pass frame or access it differently

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points = [(x, y)]
        selecting_roi = True
        roi_selected = False

    elif event == cv2.EVENT_LBUTTONUP:
        roi_points.append((x, y))
        selecting_roi = False
        roi_selected = True
        # Now you have the two points for the ROI: roi_points[0] and roi_points[1]

    elif event == cv2.EVENT_MOUSEMOVE and selecting_roi:
        # Optional: Draw a rectangle as the user drags
        temp_frame = frame.copy() # Work on a copy to avoid persistent drawing
        cv2.rectangle(temp_frame, roi_points[0], (x, y), (0, 255, 0), 2)
        cv2.imshow("Object Tracking with ROI Selection", temp_frame)

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

    # Inside your main() function, before the while loop:
    cv2.namedWindow("Object Tracking with ROI Selection")
    cv2.setMouseCallback("Object Tracking with ROI Selection", select_roi_callback)

    # Add a loop to handle ROI selection
    print("Select ROI by clicking and dragging a rectangle in the window.")
    while not roi_selected:
        # Read a frame from the camera (you already have this logic)
        frames = pipeline.wait_for_frames(100)
        if frames is None:
            continue
        color_frame = frames.get_color_frame()
        if color_frame is None:
            continue
        frame = frame_to_bgr_image(color_frame)
        if frame is None:
            print("failed to convert frame to image")
            continue
        frame = cv2.GaussianBlur(frame, (3, 3), 0)

        # Display the frame for ROI selection
        # If selecting_roi is True, draw the current rectangle
        # if selecting_roi and len(roi_points) > 0:
             # temp_frame = frame.copy()
             # cv2.rectangle(temp_frame, roi_points[0], (x, y), (0, 255, 0), 2)
             # cv2.imshow("Object Tracking with ROI Selection", temp_frame)
        # else:
        cv2.imshow("Object Tracking with ROI Selection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break # Exit if user presses 'q' during selection

    # Once roi_selected is True, calculate the ROI coordinates and set it
    if roi_selected:
        # Ensure the points are in top-left, bottom-right order
        p1 = roi_points[0]
        p2 = roi_points[1]
        x_roi = min(p1[0], p2[0])
        y_roi = min(p1[1], p2[1])
        w_roi = abs(p1[0] - p2[0])
        h_roi = abs(p1[1] - p2[1])
        tracker.set_roi(x_roi, y_roi, w_roi, h_roi)
        print(f"ROI set to: ({x_roi}, {y_roi}, {w_roi}, {h_roi})")

        # Create a new window for the selected ROI
        cv2.namedWindow("Selected ROI")

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

            # Initialize variables for drawing
            center_orig = None
            bbox_orig = None

            # Detect object (operates within ROI if set, returns coords in original frame)
            if tracker.roi:
                 center_orig, bbox_orig = tracker.detect_object(frame)
                 # Get ROI coordinates for cropping
                 x_roi, y_roi, w_roi, h_roi = tracker.roi
                 roi_frame = frame[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi].copy() # Use .copy() to avoid modifying original frame

            # Draw on the original frame (if object detected)
            if center_orig is not None and bbox_orig is not None:
                # Update trajectory with original frame coordinates
                tracker.update_trajectory(center_orig)

                # Draw bounding box on original frame
                x_orig, y_orig, w_orig, h_orig = bbox_orig
                cv2.rectangle(frame, (x_orig, y_orig), (x_orig + w_orig, y_orig + h_orig), (0, 255, 255), 2)  # Yellow box

                # Draw current position on original frame
                cv2.circle(frame, center_orig, 5, (0, 255, 255), -1)  # Yellow dot

                # Draw trajectory on original frame
                tracker.draw_trajectory(frame)

                # Add current position text on original frame
                cv2.putText(frame, f"Position: ({center_orig[0]}, {center_orig[1]})",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Draw on the ROI frame (if object detected within ROI)
                if tracker.roi:
                    # Calculate bbox and center relative to the ROI frame
                    x_roi_obj = x_orig - x_roi
                    y_roi_obj = y_orig - y_roi
                    center_x_roi = center_orig[0] - x_roi
                    center_y_roi = center_orig[1] - y_roi

                    # Draw bounding box on ROI frame
                    cv2.rectangle(roi_frame, (x_roi_obj, y_roi_obj), (x_roi_obj + w_orig, y_roi_obj + h_orig), (0, 255, 255), 2)

                    # Draw current position on ROI frame
                    cv2.circle(roi_frame, (center_x_roi, center_y_roi), 5, (0, 255, 255), -1)

                    # Add current position text on ROI frame (relative to ROI)
                    cv2.putText(roi_frame, f"Position: ({center_x_roi}, {center_y_roi})",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    # Draw trajectory on ROI frame
                    # Pass the roi_frame and the ROI's top-left corner coordinates
                    tracker.draw_trajectory_on_roi(roi_frame, x_roi, y_roi)

            # Show the original frame
            cv2.imshow("Object Tracking with ROI Selection", frame)

            # Show the ROI frame (resized)
            if tracker.roi:
                 # Define desired size for the ROI window (e.g., double the original size)
                 new_width = w_roi * 2
                 new_height = h_roi * 2
                 resized_roi_frame = cv2.resize(roi_frame, (new_width, new_height))
                 cv2.imshow("Selected ROI", resized_roi_frame)

            # Break loop on 'q' press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        except KeyboardInterrupt:
            break

    cv2.destroyAllWindows()
    pipeline.stop()

if __name__ == "__main__":
    main() 