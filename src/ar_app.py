"""
Main Augmented Reality Application
Real-time AR marker detection and 3D model projection

This application combines all AR components:
- Real-time camera capture
- Marker detection and tracking
- 3D model rendering
- User interface and controls
"""

import cv2
import numpy as np
import os
import sys
import argparse
from typing import Dict, List, Optional, Tuple

# Import our AR modules
from ar_engine import AREngine
from renderer_3d import Renderer3D


class ARApplication:
    """
    Main AR Application class that handles real-time processing
    and user interaction.
    """
    
    def __init__(self, camera_id: int = 0, detector_type: str = 'ORB'):
        """
        Initialize the AR application.
        
        Args:
            camera_id: Camera device ID (usually 0 for default camera)
            detector_type: Feature detector type ('ORB', 'SIFT', 'SURF')
        """
        self.camera_id = camera_id
        self.detector_type = detector_type
        
        # Initialize AR components
        self.ar_engine = AREngine(detector_type=detector_type)
        self.renderer_3d = Renderer3D()
        
        # Camera and video capture
        self.cap = None
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # Application state
        self.is_running = False
        self.current_model = 'cube'  # 'cube', 'pyramid', 'axes'
        self.show_keypoints = False
        self.show_fps = True
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_timer = cv2.getTickCount()
        self.current_fps = 0
        
        # Marker database
        self.markers = {}
        
    def initialize_camera(self) -> bool:
        """
        Initialize camera capture.
        
        Returns:
            True if camera initialized successfully, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                print(f"Error: Could not open camera {self.camera_id}")
                return False
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Get actual camera properties
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            print(f"Camera initialized: {width}x{height} @ {fps} FPS")
            
            # Set default camera matrix based on resolution
            self.camera_matrix = np.array([
                [width * 0.8, 0, width / 2],
                [0, width * 0.8, height / 2],
                [0, 0, 1]
            ], dtype=np.float32)
            
            self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)
            self.renderer_3d.set_camera_parameters(self.camera_matrix, self.dist_coeffs)
            
            return True
            
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def load_markers(self, markers_dir: str = "../markers") -> int:
        """
        Load AR markers from directory.
        
        Args:
            markers_dir: Directory containing marker images
            
        Returns:
            Number of markers loaded
        """
        markers_loaded = 0
        
        # Get absolute path
        if not os.path.isabs(markers_dir):
            markers_dir = os.path.join(os.path.dirname(__file__), markers_dir)
        
        if not os.path.exists(markers_dir):
            print(f"Markers directory not found: {markers_dir}")
            return 0
        
        # Load all PNG and JPG files as markers
        for filename in os.listdir(markers_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                marker_path = os.path.join(markers_dir, filename)
                marker_id = os.path.splitext(filename)[0]
                
                try:
                    self.ar_engine.add_reference_marker(marker_id, marker_path)
                    self.markers[marker_id] = marker_path
                    markers_loaded += 1
                    print(f"Loaded marker: {marker_id}")
                except Exception as e:
                    print(f"Error loading marker {filename}: {e}")
        
        print(f"Total markers loaded: {markers_loaded}")
        return markers_loaded
    
    def update_fps(self):
        """Update FPS counter."""
        self.fps_counter += 1
        if self.fps_counter >= 30:  # Update every 30 frames
            current_time = cv2.getTickCount()
            time_diff = (current_time - self.fps_timer) / cv2.getTickFrequency()
            self.current_fps = self.fps_counter / time_diff
            self.fps_counter = 0
            self.fps_timer = current_time
    
    def draw_ui(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw user interface elements on the frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with UI elements
        """
        result_frame = frame.copy()
        
        # Draw FPS
        if self.show_fps:
            fps_text = f"FPS: {self.current_fps:.1f}"
            cv2.putText(result_frame, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw current model type
        model_text = f"Model: {self.current_model}"
        cv2.putText(result_frame, model_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw detector type
        detector_text = f"Detector: {self.detector_type}"
        cv2.putText(result_frame, detector_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw instructions
        instructions = [
            "Controls:",
            "1,2,3 - Switch models",
            "k - Toggle keypoints",
            "f - Toggle FPS",
            "q - Quit"
        ]
        
        y_offset = result_frame.shape[0] - len(instructions) * 25 - 10
        for i, instruction in enumerate(instructions):
            cv2.putText(result_frame, instruction, (10, y_offset + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result_frame
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame for AR detection and rendering.
        
        Args:
            frame: Input camera frame
            
        Returns:
            Processed frame with AR content
        """
        result_frame = frame.copy()
        
        # Try to detect each loaded marker
        for marker_id in self.markers.keys():
            try:
                # Process frame with AR engine
                processed_frame, detection_success, homography = self.ar_engine.process_frame(frame, marker_id)
                
                if detection_success and homography is not None:
                    # Render 3D content
                    result_frame = self.renderer_3d.render_ar_scene(
                        processed_frame, homography, 
                        model_type=self.current_model, 
                        marker_size=1.0
                    )
                    break  # Only render for the first detected marker
                    
            except Exception as e:
                print(f"Error processing marker {marker_id}: {e}")
                continue
        
        return result_frame
    
    def handle_keyboard_input(self, key: int) -> bool:
        """
        Handle keyboard input for application controls.
        
        Args:
            key: Key code from cv2.waitKey()
            
        Returns:
            True to continue running, False to quit
        """
        key = key & 0xFF
        
        if key == ord('q') or key == 27:  # 'q' or ESC
            return False
        
        elif key == ord('1'):
            self.current_model = 'cube'
            print("Switched to cube model")
        
        elif key == ord('2'):
            self.current_model = 'pyramid'
            print("Switched to pyramid model")
        
        elif key == ord('3'):
            self.current_model = 'axes'
            print("Switched to axes model")
        
        elif key == ord('k'):
            self.show_keypoints = not self.show_keypoints
            print(f"Keypoints display: {'ON' if self.show_keypoints else 'OFF'}")
        
        elif key == ord('f'):
            self.show_fps = not self.show_fps
            print(f"FPS display: {'ON' if self.show_fps else 'OFF'}")
        
        return True
    
    def run(self) -> None:
        """
        Main application loop.
        """
        print("Starting AR Application...")
        
        # Initialize camera
        if not self.initialize_camera():
            print("Failed to initialize camera. Exiting.")
            return
        
        # Load markers
        markers_count = self.load_markers()
        if markers_count == 0:
            print("No markers loaded. Exiting.")
            return
        
        print(f"AR Application ready with {markers_count} markers!")
        print("Press 'q' to quit, '1/2/3' to switch models, 'k' for keypoints, 'f' for FPS")
        
        self.is_running = True
        
        try:
            while self.is_running:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Could not read frame from camera")
                    break
                
                # Process frame for AR
                result_frame = self.process_frame(frame)
                
                # Add UI elements
                result_frame = self.draw_ui(result_frame)
                
                # Update FPS
                self.update_fps()
                
                # Display result
                cv2.imshow('Augmented Reality', result_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1)
                if key != -1:
                    if not self.handle_keyboard_input(key):
                        break
        
        except KeyboardInterrupt:
            print("\nApplication interrupted by user")
        
        except Exception as e:
            print(f"Error in main loop: {e}")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up...")
        
        if self.cap is not None:
            self.cap.release()
        
        cv2.destroyAllWindows()
        print("AR Application closed.")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Augmented Reality Application')
    parser.add_argument('--camera', type=int, default=0, 
                       help='Camera device ID (default: 0)')
    parser.add_argument('--detector', type=str, default='ORB', 
                       choices=['ORB', 'SIFT', 'SURF'],
                       help='Feature detector type (default: ORB)')
    parser.add_argument('--markers', type=str, default='../markers',
                       help='Path to markers directory (default: ../markers)')
    
    args = parser.parse_args()
    
    # Create and run AR application
    app = ARApplication(camera_id=args.camera, detector_type=args.detector)
    
    # Override markers directory if specified
    if args.markers != '../markers':
        app.markers_dir = args.markers
    
    app.run()


if __name__ == "__main__":
    main()

