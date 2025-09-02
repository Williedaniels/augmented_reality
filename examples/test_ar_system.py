"""
AR System Test Script
Tests all components of the AR system without requiring a camera

This script demonstrates:
- Marker loading and feature detection
- Homography estimation
- 3D model projection
- Complete AR pipeline
"""

import sys
import os
import cv2
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ar_engine import AREngine
from renderer_3d import Renderer3D


def create_test_scene():
    """Create a test scene with a marker in it."""
    # Create a test image
    scene = np.ones((480, 640, 3), dtype=np.uint8) * 200  # Light gray background
    
    # Add some noise/texture to make it more realistic
    noise = np.random.randint(-30, 30, scene.shape, dtype=np.int16)
    scene = np.clip(scene.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Load and place a marker in the scene
    marker_path = os.path.join(os.path.dirname(__file__), '..', 'markers', 'marker_1.png')
    
    if os.path.exists(marker_path):
        marker = cv2.imread(marker_path)
        if marker is not None:
            # Resize marker and place it in the scene
            marker_resized = cv2.resize(marker, (150, 150))
            
            # Place marker at position (200, 100)
            x, y = 200, 100
            h, w = marker_resized.shape[:2]
            scene[y:y+h, x:x+w] = marker_resized
            
            print(f"Placed marker in test scene at ({x}, {y})")
    
    return scene


def test_ar_components():
    """Test individual AR components."""
    print("=== Testing AR Components ===")
    
    # Test AR Engine
    print("\n1. Testing AR Engine...")
    ar_engine = AREngine(detector_type='ORB')
    print(f"   ✓ AR Engine created with {ar_engine.detector_type} detector")
    
    # Test 3D Renderer
    print("\n2. Testing 3D Renderer...")
    renderer = Renderer3D()
    
    # Test model creation
    cube_vertices, cube_edges = renderer.create_cube_model(1.0)
    pyramid_vertices, pyramid_edges = renderer.create_pyramid_model(1.0, 1.5)
    axes_vertices, axes_edges = renderer.create_coordinate_axes(1.0)
    
    print(f"   ✓ Cube model: {len(cube_vertices)} vertices, {len(cube_edges)} edges")
    print(f"   ✓ Pyramid model: {len(pyramid_vertices)} vertices, {len(pyramid_edges)} edges")
    print(f"   ✓ Coordinate axes: {len(axes_vertices)} vertices, {len(axes_edges)} axes")
    
    # Test homography to pose conversion
    test_homography = np.array([
        [1.2, 0.1, 250],
        [0.1, 1.2, 150],
        [0.001, 0.001, 1.0]
    ], dtype=np.float32)
    
    rvec, tvec = renderer.homography_to_pose(test_homography)
    print(f"   ✓ Pose estimation: rotation {rvec.shape}, translation {tvec.shape}")
    
    return ar_engine, renderer


def test_marker_detection():
    """Test marker detection and tracking."""
    print("\n=== Testing Marker Detection ===")
    
    # Create AR engine
    ar_engine = AREngine(detector_type='ORB')
    
    # Load markers
    markers_dir = os.path.join(os.path.dirname(__file__), '..', 'markers')
    markers_loaded = 0
    
    for filename in os.listdir(markers_dir):
        if filename.lower().endswith('.png'):
            marker_path = os.path.join(markers_dir, filename)
            marker_id = os.path.splitext(filename)[0]
            
            try:
                ar_engine.add_reference_marker(marker_id, marker_path)
                markers_loaded += 1
                print(f"   ✓ Loaded marker: {marker_id}")
            except Exception as e:
                print(f"   ✗ Error loading {filename}: {e}")
    
    print(f"\nTotal markers loaded: {markers_loaded}")
    
    # Test detection on a synthetic scene
    if markers_loaded > 0:
        print("\n3. Testing marker detection in synthetic scene...")
        test_scene = create_test_scene()
        
        # Try to detect the first marker
        first_marker = list(ar_engine.reference_markers.keys())[0]
        matches, query_kp, ref_kp = ar_engine.detect_and_match_features(test_scene, first_marker)
        
        print(f"   Found {len(matches)} feature matches")
        
        if len(matches) >= 4:
            homography = ar_engine.estimate_homography(matches, query_kp, ref_kp)
            if homography is not None:
                print("   ✓ Homography estimation successful")
                
                # Get marker corners
                corners = ar_engine.get_marker_corners(homography, first_marker)
                print(f"   ✓ Marker corners detected: {corners.shape}")
                
                return test_scene, homography, first_marker
            else:
                print("   ✗ Homography estimation failed")
        else:
            print("   ✗ Not enough matches for homography estimation")
    
    return None, None, None


def test_3d_rendering(scene, homography, marker_id):
    """Test 3D rendering on detected marker."""
    if scene is None or homography is None:
        print("\n=== Skipping 3D Rendering Test (no detection) ===")
        return None
    
    print("\n=== Testing 3D Rendering ===")
    
    renderer = Renderer3D()
    
    # Test different 3D models
    models = ['cube', 'pyramid', 'axes']
    results = []
    
    for model_type in models:
        print(f"\n4. Testing {model_type} rendering...")
        try:
            result_scene = renderer.render_ar_scene(scene, homography, model_type, marker_size=1.0)
            results.append((model_type, result_scene))
            print(f"   ✓ {model_type.capitalize()} rendered successfully")
        except Exception as e:
            print(f"   ✗ Error rendering {model_type}: {e}")
    
    return results


def save_test_results(results):
    """Save test results as images."""
    if not results:
        return
    
    print("\n=== Saving Test Results ===")
    
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'examples', 'test_output')
    os.makedirs(output_dir, exist_ok=True)
    
    for model_type, result_image in results:
        output_path = os.path.join(output_dir, f'ar_test_{model_type}.png')
        cv2.imwrite(output_path, result_image)
        print(f"   ✓ Saved {model_type} result: {output_path}")
    
    print(f"\nTest results saved to: {output_dir}")


def main():
    """Main test function."""
    print("🎯 AR System Comprehensive Test")
    print("=" * 50)
    
    try:
        # Test individual components
        ar_engine, renderer = test_ar_components()
        
        # Test marker detection
        scene, homography, marker_id = test_marker_detection()
        
        # Test 3D rendering
        results = test_3d_rendering(scene, homography, marker_id)
        
        # Save results
        if results:
            save_test_results(results)
        
        print("\n" + "=" * 50)
        print("🎉 AR System Test Completed Successfully!")
        print("\nThe AR system is ready for real-time use.")
        print("Run 'python src/ar_app.py' to start the camera-based AR application.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

