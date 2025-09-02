"""
3D Renderer for Augmented Reality
Implements 3D projection mathematics and rendering pipeline

This module handles:
- Camera calibration and intrinsic parameters
- 3D to 2D projection using homography
- 3D model rendering and visualization
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict, Any


class Renderer3D:
    """
    3D Renderer class that handles projection of 3D models onto 2D images
    using camera calibration and homography transformations.
    """
    
    def __init__(self, camera_matrix: Optional[np.ndarray] = None, 
                 dist_coeffs: Optional[np.ndarray] = None):
        """
        Initialize the 3D renderer.
        
        Args:
            camera_matrix: Camera intrinsic matrix (3x3)
            dist_coeffs: Distortion coefficients
        """
        # Default camera parameters (will be updated with calibration)
        if camera_matrix is None:
            # Default camera matrix for typical webcam
            self.camera_matrix = np.array([
                [800, 0, 320],
                [0, 800, 240],
                [0, 0, 1]
            ], dtype=np.float32)
        else:
            self.camera_matrix = camera_matrix
            
        if dist_coeffs is None:
            self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        else:
            self.dist_coeffs = dist_coeffs
    
    def set_camera_parameters(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
        """Set camera calibration parameters."""
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
    
    def homography_to_pose(self, homography: np.ndarray, marker_size: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose homography matrix to get camera pose (rotation and translation).
        
        Args:
            homography: 3x3 homography matrix
            marker_size: Physical size of the marker in world units
            
        Returns:
            Tuple of (rotation_vector, translation_vector)
        """
        # Normalize homography
        h = homography / np.linalg.norm(homography[:, 0])
        
        # Extract rotation and translation from homography
        # H = K * [r1 r2 t] where r1, r2 are first two columns of rotation matrix
        h1 = h[:, 0]
        h2 = h[:, 1]
        h3 = h[:, 2]
        
        # Compute rotation vectors
        r1 = np.linalg.inv(self.camera_matrix) @ h1
        r2 = np.linalg.inv(self.camera_matrix) @ h2
        t = np.linalg.inv(self.camera_matrix) @ h3
        
        # Normalize rotation vectors
        r1 = r1 / np.linalg.norm(r1)
        r2 = r2 / np.linalg.norm(r2)
        
        # Compute third rotation vector (cross product)
        r3 = np.cross(r1, r2)
        
        # Create rotation matrix
        rotation_matrix = np.column_stack((r1, r2, r3))
        
        # Ensure proper rotation matrix (orthogonal with determinant 1)
        U, _, Vt = np.linalg.svd(rotation_matrix)
        rotation_matrix = U @ Vt
        
        # Convert rotation matrix to rotation vector
        rotation_vector, _ = cv2.Rodrigues(rotation_matrix)
        
        # Scale translation by marker size
        translation_vector = t * marker_size
        
        return rotation_vector, translation_vector
    
    def project_3d_points(self, points_3d: np.ndarray, rotation_vector: np.ndarray, 
                         translation_vector: np.ndarray) -> np.ndarray:
        """
        Project 3D points to 2D image coordinates.
        
        Args:
            points_3d: 3D points in world coordinates (Nx3)
            rotation_vector: Camera rotation vector
            translation_vector: Camera translation vector
            
        Returns:
            2D projected points (Nx2)
        """
        # Use OpenCV's projectPoints function
        points_2d, _ = cv2.projectPoints(
            points_3d.reshape(-1, 1, 3),
            rotation_vector,
            translation_vector,
            self.camera_matrix,
            self.dist_coeffs
        )
        
        return points_2d.reshape(-1, 2)
    
    def create_cube_model(self, size: float = 1.0) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Create a 3D cube model with vertices and edges.
        
        Args:
            size: Size of the cube
            
        Returns:
            Tuple of (vertices, edges)
        """
        # Define cube vertices
        vertices = np.array([
            [0, 0, 0],          # 0: bottom-front-left
            [size, 0, 0],       # 1: bottom-front-right
            [size, size, 0],    # 2: bottom-back-right
            [0, size, 0],       # 3: bottom-back-left
            [0, 0, -size],      # 4: top-front-left
            [size, 0, -size],   # 5: top-front-right
            [size, size, -size], # 6: top-back-right
            [0, size, -size]    # 7: top-back-left
        ], dtype=np.float32)
        
        # Define cube edges (connections between vertices)
        edges = [
            # Bottom face
            (0, 1), (1, 2), (2, 3), (3, 0),
            # Top face
            (4, 5), (5, 6), (6, 7), (7, 4),
            # Vertical edges
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        
        return vertices, edges
    
    def create_pyramid_model(self, base_size: float = 1.0, height: float = 1.0) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Create a 3D pyramid model.
        
        Args:
            base_size: Size of the pyramid base
            height: Height of the pyramid
            
        Returns:
            Tuple of (vertices, edges)
        """
        # Define pyramid vertices
        vertices = np.array([
            [0, 0, 0],                    # 0: base corner 1
            [base_size, 0, 0],            # 1: base corner 2
            [base_size, base_size, 0],    # 2: base corner 3
            [0, base_size, 0],            # 3: base corner 4
            [base_size/2, base_size/2, -height]  # 4: apex
        ], dtype=np.float32)
        
        # Define pyramid edges
        edges = [
            # Base edges
            (0, 1), (1, 2), (2, 3), (3, 0),
            # Edges to apex
            (0, 4), (1, 4), (2, 4), (3, 4)
        ]
        
        return vertices, edges
    
    def create_coordinate_axes(self, length: float = 1.0) -> Tuple[np.ndarray, List[Tuple[int, int, Tuple[int, int, int]]]]:
        """
        Create 3D coordinate axes for visualization.
        
        Args:
            length: Length of each axis
            
        Returns:
            Tuple of (vertices, colored_edges)
        """
        # Define axis vertices
        vertices = np.array([
            [0, 0, 0],        # 0: origin
            [length, 0, 0],   # 1: X-axis end (red)
            [0, length, 0],   # 2: Y-axis end (green)
            [0, 0, -length]   # 3: Z-axis end (blue)
        ], dtype=np.float32)
        
        # Define colored edges (vertex1, vertex2, color_BGR)
        colored_edges = [
            (0, 1, (0, 0, 255)),    # X-axis: red
            (0, 2, (0, 255, 0)),    # Y-axis: green
            (0, 3, (255, 0, 0))     # Z-axis: blue
        ]
        
        return vertices, colored_edges
    
    def draw_3d_model(self, image: np.ndarray, vertices: np.ndarray, edges: List[Tuple[int, int]], 
                     rotation_vector: np.ndarray, translation_vector: np.ndarray,
                     color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """
        Draw a 3D model on the image.
        
        Args:
            image: Input image
            vertices: 3D model vertices
            edges: Model edges (vertex index pairs)
            rotation_vector: Camera rotation vector
            translation_vector: Camera translation vector
            color: Line color (BGR)
            thickness: Line thickness
            
        Returns:
            Image with 3D model drawn
        """
        # Project 3D vertices to 2D
        points_2d = self.project_3d_points(vertices, rotation_vector, translation_vector)
        
        # Draw edges
        result_image = image.copy()
        for edge in edges:
            pt1 = tuple(map(int, points_2d[edge[0]]))
            pt2 = tuple(map(int, points_2d[edge[1]]))
            cv2.line(result_image, pt1, pt2, color, thickness)
        
        return result_image
    
    def draw_coordinate_axes(self, image: np.ndarray, rotation_vector: np.ndarray, 
                           translation_vector: np.ndarray, length: float = 1.0, 
                           thickness: int = 3) -> np.ndarray:
        """
        Draw 3D coordinate axes on the image.
        
        Args:
            image: Input image
            rotation_vector: Camera rotation vector
            translation_vector: Camera translation vector
            length: Length of axes
            thickness: Line thickness
            
        Returns:
            Image with coordinate axes drawn
        """
        vertices, colored_edges = self.create_coordinate_axes(length)
        
        # Project 3D vertices to 2D
        points_2d = self.project_3d_points(vertices, rotation_vector, translation_vector)
        
        # Draw colored axes
        result_image = image.copy()
        for edge in colored_edges:
            pt1 = tuple(map(int, points_2d[edge[0]]))
            pt2 = tuple(map(int, points_2d[edge[1]]))
            color = edge[2]
            cv2.line(result_image, pt1, pt2, color, thickness)
        
        # Add axis labels
        if len(points_2d) >= 4:
            cv2.putText(result_image, 'X', tuple(map(int, points_2d[1])), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(result_image, 'Y', tuple(map(int, points_2d[2])), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(result_image, 'Z', tuple(map(int, points_2d[3])), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return result_image
    
    def render_ar_scene(self, image: np.ndarray, homography: np.ndarray, 
                       model_type: str = 'cube', marker_size: float = 1.0) -> np.ndarray:
        """
        Render a complete AR scene with 3D models.
        
        Args:
            image: Input image
            homography: Homography matrix from marker detection
            model_type: Type of 3D model to render ('cube', 'pyramid', 'axes')
            marker_size: Physical size of the marker
            
        Returns:
            Image with AR scene rendered
        """
        # Get camera pose from homography
        rotation_vector, translation_vector = self.homography_to_pose(homography, marker_size)
        
        result_image = image.copy()
        
        # Render coordinate axes
        result_image = self.draw_coordinate_axes(result_image, rotation_vector, 
                                               translation_vector, marker_size * 0.5)
        
        # Render 3D model based on type
        if model_type == 'cube':
            vertices, edges = self.create_cube_model(marker_size * 0.5)
            result_image = self.draw_3d_model(result_image, vertices, edges, 
                                            rotation_vector, translation_vector, 
                                            color=(0, 255, 255), thickness=3)
        
        elif model_type == 'pyramid':
            vertices, edges = self.create_pyramid_model(marker_size * 0.5, marker_size * 0.7)
            result_image = self.draw_3d_model(result_image, vertices, edges, 
                                            rotation_vector, translation_vector, 
                                            color=(255, 0, 255), thickness=3)
        
        return result_image


def test_renderer_3d():
    """Test function for the 3D renderer."""
    print("Testing 3D Renderer...")
    
    # Create renderer
    renderer = Renderer3D()
    
    # Test model creation
    cube_vertices, cube_edges = renderer.create_cube_model(1.0)
    pyramid_vertices, pyramid_edges = renderer.create_pyramid_model(1.0, 1.5)
    axes_vertices, axes_edges = renderer.create_coordinate_axes(1.0)
    
    print(f"Cube model: {len(cube_vertices)} vertices, {len(cube_edges)} edges")
    print(f"Pyramid model: {len(pyramid_vertices)} vertices, {len(pyramid_edges)} edges")
    print(f"Coordinate axes: {len(axes_vertices)} vertices, {len(axes_edges)} axes")
    
    # Test homography to pose conversion
    test_homography = np.array([
        [1.0, 0.1, 100],
        [0.1, 1.0, 100],
        [0.001, 0.001, 1.0]
    ], dtype=np.float32)
    
    rvec, tvec = renderer.homography_to_pose(test_homography)
    print(f"Rotation vector shape: {rvec.shape}")
    print(f"Translation vector shape: {tvec.shape}")
    
    print("3D Renderer initialized successfully!")
    
    return renderer


if __name__ == "__main__":
    test_renderer_3d()

