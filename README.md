# AI-Assisted SmartBrake: Automatic Braking System using YOLOv5 and OpenCV

AI-Assisted SmartBrake is an innovative Automatic Braking System that harnesses the power of YOLOv5 and OpenCV to provide advanced object detection and collision avoidance capabilities. This cutting-edge system is designed to enhance road safety and prevent accidents by detecting potential obstacles, pedestrians, and vehicles in real-time.

### Features:

YOLOv5 Object Detection: Utilizes the state-of-the-art YOLOv5 model for robust and accurate object detection.
Real-Time Collision Avoidance: Instantly activates the vehicle's braking system upon detecting potential collisions.
High-Performance OpenCV: Leverages the power of OpenCV for efficient image processing and analysis.
Customizable Thresholds: Offers adjustable sensitivity levels for personalized safety preferences.
Seamless Integration: Easily integrated into autonomous vehicles and robotic applications.

### How It Works:

- YOLOv5 Model: The system employs YOLOv5, a deep learning-based object detection model, to identify objects in the vehicle's path.
- Real-Time Detection: Using live video feed from cameras, the system performs real-time object detection for timely decision-making.
- Collision Avoidance: Upon identifying potential obstacles, SmartBrake automatically engages the brakes to prevent collisions.
- Customizable Safety: Users can fine-tune the system's sensitivity to suit different driving conditions.


### Prerequisites

- Python 3.x
- OpenCV
- PyTorch
- NumPy

### Installation

1. Clone this repository.
2. Install the required dependencies

```bash
pip3 install torch opencv numpy
```

### Usage

1. Download pre-trained YOLOv5 weights or train your own model.
2. Provide the path to the YOLOv5 weights in the code.
3. Run the script with the video file.
4. View the object detection results and Bird's Eye View visualization.

For more detailed usage instructions and options, refer to the project documentation.

### Run

```bash
python3 yoloV5_sim.py
```

### Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.
