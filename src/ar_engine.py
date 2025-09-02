"""
Augmented Reality Engine
Built from scratch following build-your-own-x guidelines

This module implements the core AR functionality including:
- Feature detection and matching
- Homography estimation
- 3D projection and rendering
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict, Any


class AREngine:
    """
    Main Augmented Reality Engine class that handles marker detection,
    tracking, and 3D model projection.
    """
    
    def __init__(self, detector_type: str = 'ORB'):
        """
        Initialize the AR Engine.
        
        Args:
            detector_type: Feature detector type ('ORB', 'SIFT', 'SURF')
        """
        self.detector_type = detector_type
        self.detector = self._create_detector(detector_type)
        self.matcher = self._create_matcher()
        
        # Camera calibration parameters (will be set during calibration)
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # Reference marker data
        self.reference_markers = {}
        
        # 3D model vertices for rendering
        self.model_vertices = self._create_default_3d_model()
        
    def _create_detector(self, detector_type: str):
        """Create feature detector based on type."""
        if detector_type == 'ORB':
            return cv2.ORB_create(nfeatures=1000)
        elif detector_type == 'SIFT':
            return cv2.SIFT_create()
        elif detector_type == 'SURF':
            return cv2.xfeatures2d.SURF_create(400)
        else:
            raise ValueError(f"Unsupported detector type: {detector_type}")
    
    def _create_matcher(self):
        """Create feature matcher."""
        if self.detector_type == 'ORB':
            # Use Hamming distance for binary descriptors
            return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            # Use L2 distance for float descriptors
            return cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    def _create_default_3d_model(self) -> np.ndarray:
        """Create a default 3D cube model for rendering."""
        # Define a 3D cube with vertices
        vertices = np.array([
            [0, 0, 0],      # Bottom face
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, -1],     # Top face
            [1, 0, -1],
            [1, 1, -1],
            [0, 1, -1]
        ], dtype=np.float32)
        
        return vertices
    
    def add_reference_marker(self, marker_id: str, image_path: str):
        """
        Add a reference marker image for tracking.
        
        Args:
            marker_id: Unique identifier for the marker
            image_path: Path to the reference marker image
        """
        # Load reference image
        ref_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if ref_image is None:
            raise ValueError(f"Could not load reference image: {image_path}")
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.detector.detectAndCompute(ref_image, None)
        
        # Store reference data
        self.reference_markers[marker_id] = {
            'image': ref_image,
            'keypoints': keypoints,
            'descriptors': descriptors,
            'shape': ref_image.shape
        }
        
        print(f"Added reference marker '{marker_id}' with {len(keypoints)} keypoints")
    
    def detect_and_match_features(self, query_image: np.ndarray, marker_id: str) -> Tuple[List, List, List]:
        """
        Detect features in query image and match with reference marker.
        
        Args:
            query_image: Input image to search for markers
            marker_id: ID of the reference marker to match against
            
        Returns:
            Tuple of (good_matches, query_keypoints, reference_keypoints)
        """
        if marker_id not in self.reference_markers:
            raise ValueError(f"Reference marker '{marker_id}' not found")
        
        # Convert to grayscale if needed
        if len(query_image.shape) == 3:
            query_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
        else:
            query_gray = query_image
        
        # Detect keypoints and compute descriptors in query image
        query_kp, query_desc = self.detector.detectAndCompute(query_gray, None)
        
        if query_desc is None or len(query_desc) == 0:
            return [], [], []
        
        # Get reference data
        ref_data = self.reference_markers[marker_id]
        ref_desc = ref_data['descriptors']
        
        if ref_desc is None or len(ref_desc) == 0:
            return [], [], []
        
        # Match descriptors
        matches = self.matcher.match(query_desc, ref_desc)
        
        # Sort matches by distance (best matches first)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Filter good matches using distance threshold
        good_matches = []
        if len(matches) > 0:
            # Use adaptive threshold based on match quality
            distance_threshold = min(50, matches[0].distance * 2.5)
            good_matches = [m for m in matches if m.distance < distance_threshold]
        
        return good_matches, query_kp, ref_data['keypoints']
    
    def estimate_homography(self, good_matches: List, query_kp: List, ref_kp: List) -> Optional[np.ndarray]:
        """
        Estimate homography matrix from matched keypoints using RANSAC.
        
        Args:
            good_matches: List of good feature matches
            query_kp: Query image keypoints
            ref_kp: Reference image keypoints
            
        Returns:
            Homography matrix or None if estimation fails
        """
        if len(good_matches) < 4:
            return None
        
        # Extract matched point coordinates
        src_pts = np.float32([ref_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([query_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Estimate homography using RANSAC
        homography, mask = cv2.findHomography(
            src_pts, dst_pts, 
            cv2.RANSAC, 
            ransacReprojThreshold=5.0,
            confidence=0.99,
            maxIters=2000
        )
        
        return homography
    
    def get_marker_corners(self, homography: np.ndarray, marker_id: str) -> np.ndarray:
        """
        Get the corners of the detected marker in the query image.
        
        Args:
            homography: Homography matrix
            marker_id: Reference marker ID
            
        Returns:
            Array of corner points in query image coordinates
        """
        if marker_id not in self.reference_markers:
            raise ValueError(f"Reference marker '{marker_id}' not found")
        
        # Get reference image dimensions
        h, w = self.reference_markers[marker_id]['shape']
        
        # Define corners of reference image
        ref_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        
        # Transform corners to query image coordinates
        query_corners = cv2.perspectiveTransform(ref_corners, homography)
        
        return query_corners
    
    def draw_detection_results(self, image: np.ndarray, corners: np.ndarray, 
                             matches: List = None, query_kp: List = None, 
                             ref_kp: List = None, marker_id: str = None) -> np.ndarray:
        """
        Draw detection results on the image.
        
        Args:
            image: Input image
            corners: Detected marker corners
            matches: Feature matches (optional)
            query_kp: Query keypoints (optional)
            ref_kp: Reference keypoints (optional)
            marker_id: Marker ID for reference image (optional)
            
        Returns:
            Image with detection results drawn
        """
        result_image = image.copy()
        
        # Draw marker boundary
        if corners is not None and len(corners) == 4:
            corners_int = np.int32(corners).reshape(-1, 2)
            cv2.polylines(result_image, [corners_int], True, (0, 255, 0), 3)
            
            # Add marker ID text
            if marker_id:
                cv2.putText(result_image, f"Marker: {marker_id}", 
                           tuple(corners_int[0]), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 0), 2)
        
        # Draw keypoints if available
        if query_kp:
            cv2.drawKeypoints(result_image, query_kp, result_image, 
                            color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        return result_image
    
    def process_frame(self, frame: np.ndarray, marker_id: str) -> Tuple[np.ndarray, bool, Optional[np.ndarray]]:
        """
        Process a single frame for AR marker detection.
        
        Args:
            frame: Input video frame
            marker_id: ID of marker to detect
            
        Returns:
            Tuple of (processed_frame, detection_success, homography)
        """
        # Detect and match features
        matches, query_kp, ref_kp = self.detect_and_match_features(frame, marker_id)
        
        if len(matches) < 10:  # Minimum matches required
            return frame, False, None
        
        # Estimate homography
        homography = self.estimate_homography(matches, query_kp, ref_kp)
        
        if homography is None:
            return frame, False, None
        
        # Get marker corners
        corners = self.get_marker_corners(homography, marker_id)
        
        # Draw detection results
        result_frame = self.draw_detection_results(frame, corners, matches, query_kp, ref_kp, marker_id)
        
        return result_frame, True, homography


def test_ar_engine():
    """Test function for the AR engine."""
    print("Testing AR Engine...")
    
    # Create AR engine
    ar = AREngine(detector_type='ORB')
    
    # Test with a simple image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_image, (100, 100), (300, 200), (255, 255, 255), -1)
    cv2.putText(test_image, "TEST MARKER", (120, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    print("AR Engine initialized successfully!")
    print(f"Detector type: {ar.detector_type}")
    print(f"3D model vertices shape: {ar.model_vertices.shape}")
    
    return ar


if __name__ == "__main__":
    test_ar_engine()

