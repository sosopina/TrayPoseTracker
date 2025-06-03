import pygame
import threading
import queue
import time
import math
import os

class ObjectTracker3DVisualizer:
    def __init__(self, width=1000, height=800):  # More reasonable window size
        self.width = width
        self.height = height
        self.running = True
        self.object_position = [0, 0]
        self.roi_size = (600, 450)  # Adjusted default ROI size
        
        # Initialize PyGame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("2D Plate Object Tracking Visualization")
        
        # Initialize fonts
        pygame.font.init()
        self.title_font = pygame.font.SysFont('Arial', 32, bold=True)  # Slightly smaller title
        self.info_font = pygame.font.SysFont('Arial', 20)  # Slightly smaller info text
        
        # Professional color scheme
        self.colors = {
            'background': (240, 240, 240),  # Light gray background
            'roi_fill': (30, 30, 30),       # Dark gray for ROI
            'roi_border': (100, 100, 100),  # Medium gray border
            'object': (255, 215, 0),        # Gold color for object
            'text': (50, 50, 50),           # Dark gray text
            'title': (70, 70, 70)           # Slightly lighter for title
        }
        
    def set_roi_size(self, width, height):
        """Update the ROI size"""
        # Scale up the ROI size for better visualization
        scale_factor = 1.5  # Reduced scale factor for more reasonable size
        self.roi_size = (int(width * scale_factor), int(height * scale_factor))
        
    def draw_scene(self):
        """Draw the ROI rectangle and the moving point with enhanced visuals"""
        # Clear screen with professional background color
        self.screen.fill(self.colors['background'])
        
        # Calculate rectangle position to center it
        rect_x = (self.width - self.roi_size[0]) // 2
        rect_y = (self.height - self.roi_size[1]) // 2
        
        # Draw title
        title_text = self.title_font.render("2D Plate Object Tracking", True, self.colors['title'])
        title_rect = title_text.get_rect(center=(self.width // 2, 30))
        self.screen.blit(title_text, title_rect)
        
        # Draw ROI rectangle with professional styling
        # Main rectangle
        pygame.draw.rect(self.screen, self.colors['roi_fill'], 
                        (rect_x, rect_y, self.roi_size[0], self.roi_size[1]))
        # Border with slight gradient effect
        for i in range(3):
            pygame.draw.rect(self.screen, self.colors['roi_border'], 
                           (rect_x-i, rect_y-i, self.roi_size[0]+2*i, self.roi_size[1]+2*i), 1)
        
        # Draw moving object with enhanced styling
        point_x = rect_x + self.object_position[0]
        point_y = rect_y + self.object_position[1]
        
        # Draw object with glow effect
        for radius in range(12, 8, -1):  # Slightly smaller glow
            alpha = int(255 * (1 - (radius - 8) / 4))
            glow_surface = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surface, (*self.colors['object'], alpha), (radius, radius), radius)
            self.screen.blit(glow_surface, (point_x-radius, point_y-radius))
        
        # Draw position information
        pos_text = self.info_font.render(
            f"Position: ({int(self.object_position[0])}, {int(self.object_position[1])})", 
            True, self.colors['text'])
        self.screen.blit(pos_text, (20, self.height - 30))
        
        # Draw ROI size information
        size_text = self.info_font.render(
            f"ROI Size: {self.roi_size[0]}x{self.roi_size[1]}", 
            True, self.colors['text'])
        self.screen.blit(size_text, (self.width - 180, self.height - 30))
        
        # Update display
        pygame.display.flip()
        
    def update_position(self, x, y):
        """Update the object position from the tracking system"""
        # Scale the coordinates to match the larger visualization
        scale_factor = 1.5  # Same as in set_roi_size
        x = max(0, min(x * scale_factor, self.roi_size[0]))
        y = max(0, min(y * scale_factor, self.roi_size[1]))
        
        self.object_position = [x, y]
        
    def run(self):
        """Main visualization loop"""
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
            
            # Draw the scene
            self.draw_scene()
            pygame.time.wait(10)  # Cap at ~100 FPS
            
        pygame.quit()
        
    def start(self):
        """Start the visualization in a separate thread"""
        self.thread = threading.Thread(target=self.run)
        self.thread.start()
        
    def stop(self):
        """Stop the visualization"""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()

def main():
    # Example usage
    visualizer = ObjectTracker3DVisualizer()
    visualizer.set_roi_size(400, 300)  # Example ROI size
    visualizer.start()
    
    # Simulate position updates
    try:
        while True:
            # Example: Move the object in a circle
            t = time.time()
            x = 200 + 100 * math.sin(t)  # Center at 200, radius 100
            y = 150 + 100 * math.cos(t)  # Center at 150, radius 100
            visualizer.update_position(x, y)
            time.sleep(0.016)  # ~60 FPS
    except KeyboardInterrupt:
        visualizer.stop()

if __name__ == "__main__":
    main() 