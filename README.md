# Face Recognition Speaking Detection

A web-based application that detects faces and tracks speaking duration in real-time using face recognition technology.

## Features

- Main menu with two options:
  - Start Live Stream: Real-time face and speech detection from camera
  - Analyze Recorded Video: Process pre-recorded videos for face and speech analysis
- Real-time face detection and tracking
- Speaking duration tracking for each detected person
- Modern and responsive web interface
- Support for both live video feed and recorded videos
- Automatic face recognition and tracking
- Speaking time display in minutes and seconds
- Visual representation of speaking/listening ratios with charts
- Face screenshots for each detected person
- Detailed analysis reports

## Requirements

- Python 3.7+
- OpenCV
- dlib
- face_recognition
- Flask
- Flask-SocketIO

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd SpokenDetection-AI
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Download the shape predictor file:
```bash
# The shape_predictor_68_face_landmarks.dat file should be in the root directory
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Choose one of the two options:
   - **Start Live Stream**: Begin real-time face and speech detection from your camera
   - **Analyze Recorded Video**: Upload a video file or provide a URL for analysis

4. For Live Stream:
   - Click "Start Stream" to begin detection
   - The application will detect faces and track speaking duration
   - Click "Stop Stream" to end the session and view results
   - Results include speaking times, face screenshots, and speaking/listening ratios

5. For Recorded Video:
   - Upload a video file or enter a video URL
   - Click "Start Analysis" to begin processing
   - The application will analyze the video and track speaking duration
   - Click "Stop Analysis" to end the session and view results
   - Results include speaking times, face screenshots, and speaking/listening ratios

## How it Works

1. The application uses dlib's face detector to locate faces in each frame
2. Face recognition is used to track unique individuals across frames
3. Mouth landmarks are detected to determine if a person is speaking
4. Speaking duration is tracked and updated in real-time
5. The web interface displays the video feed and speaking times
6. Results are visualized with charts showing speaking/listening ratios
7. Face screenshots are captured and displayed for each detected person

## Notes

- The application works best with good lighting conditions
- Face detection accuracy may vary depending on the camera quality and distance
- The speaking detection threshold can be adjusted in the code if needed
- For recorded videos, processing time depends on the video length and complexity

## License

This project is licensed under the MIT License - see the LICENSE file for details. 