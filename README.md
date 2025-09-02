# Build Your Own Augmented Reality

This project is a complete augmented reality system built from scratch using Python and OpenCV, following the guidelines from the build-your-own-x repository. It implements marker-based AR, allowing you to project 3D models onto real-world objects in real-time.

This project was built to understand the fundamentals of computer vision and augmented reality, and it serves as a great educational tool for anyone interested in these fields.

## Features

* **Marker-based AR**: Uses computer vision to detect and track markers in real-time.
* **Feature Detection**: Supports ORB, SIFT, and SURF for feature detection.
* **Homography Estimation**: Robustly estimates the transformation between the marker and the camera.
* **3D Model Projection**: Projects 3D models (cubes, pyramids, axes) onto the detected markers.
* **Real-time Camera Processing**: Captures video from a webcam and performs AR in real-time.
* **Customizable Markers**: You can create and use your own markers.
* **Interactive Controls**: Switch between different 3D models and toggle display options.

## How to Install and Run

This project uses Python and OpenCV. You will need Python 3 and pip installed.

1. **Clone the repository or download the source code.**
2. **Navigate to the `augmented_reality` directory.**
3. **Create and activate a virtual environment (recommended):**

   This ensures that the project's dependencies are isolated from your system's Python environment.

    ```bash
   python3 -m venv venv
   source venv/bin/activate
   # On Windows, use: venv\Scripts\activate
    ```

4. **Install the dependencies:**

   With the virtual environment activated, install the required packages.

    ```bash
   pip install -r requirements.txt
    ```

5. **Run the AR application:**

    ```bash
   python src/ar_app.py
    ```

    This will start the real-time AR application using your webcam. Point your camera at one of the markers in the `markers/` directory to see the 3D models.

6. **Run the test script (no camera needed):**

    ```bash
   python examples/test_ar_system.py
    ```

    This will run a comprehensive test of all AR components and save the output to the `examples/test_output/` directory.

## Project Structure

* `src/`: Contains the core source code for the AR system.
  * `ar_engine.py`: Implements feature detection, matching, and homography estimation.
  * `renderer_3d.py`: Implements the 3D projection and rendering pipeline.
  * `ar_app.py`: The main AR application with real-time camera processing.
* `markers/`: Contains the AR marker images.
* `models/`: (Optional) For storing 3D model files.
* `examples/`: Contains example scripts and test output.
  * `test_ar_system.py`: A comprehensive test script for the AR system.
  * `test_output/`: Contains the output images from the test script.
* `docs/`: (Optional) For additional documentation.
* `tests/`: (Optional) For unit tests.
* `requirements.txt`: The Python dependencies for the project.
* `README.md`: This file.
