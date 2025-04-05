# Gaelic Football Match Analyser

A comprehensive local LLM solution for analyzing football/Gaelic football footage to identify scoring patterns, team strategies, and player movements.

## Overview

This system processes game recordings, extracts meaningful insights, and provides strategic analysis without relying on cloud services. The solution prioritizes free/open-source tools and is designed to run on consumer hardware.

## How It Works

### 1. Overall Setup

The Gaelic Football Match Analyser consists of several interlinked components:

- **Video Processing Pipeline**: Handles raw video footage, extracts frames, detects players and the ball, and tracks movements throughout the game.
- **Computer Vision System**: Identifies players, differentiates teams based on jersey colors, detects field boundaries, and tracks ball movement.
- **Data Transformation Layer**: Converts visual data into structured data suitable for analysis.
- **Language Model Integration**: Uses fine-tuned local LLMs to analyze the structured data and generate tactical insights.
- **Visualization Interface**: Presents analysis results in an intuitive interface for coaches and analysts.

### 2. Core Functionality

#### Video Processing
- **Frame Extraction**: The system samples frames at regular intervals (default: every 30 frames) to balance performance and accuracy.
- **Player Detection**: Uses computer vision techniques (background subtraction in the current implementation) to identify moving players in each frame.
- **Team Differentiation**: Analyzes jersey colors in HSV color space to determine which team a player belongs to (team_a for blue-ish colors, team_b for red-ish colors).
- **Player Tracking**: Maintains identity of players across frames using proximity-based tracking algorithms.
- **Field Detection**: Identifies the field boundaries and key markings to provide spatial context.

#### Pass Detection
- **Ball Tracking**: Tracks the movement of the ball across frames (currently simulated but designed to be replaced with proper ball detection).
- **Pass Recognition**: Analyzes changes in ball trajectory and velocity to detect potential passes.
- **Pass Validation**: Determines if a detected pass is valid by:
  1. Finding the closest players to the ball before and after the trajectory change
  2. Confirming the players are different individuals
  3. Verifying that both players are on the same team
- **Pass Classification**: Categorizes passes as short, medium, or long based on distance.

#### Data Analysis
- **Play Segmentation**: Divides the game into discrete plays for analysis.
- **Pattern Recognition**: Identifies recurring successful patterns in team movements and formations.
- **Strategic Analysis**: Evaluates team formations and tactical approaches throughout the game.

### 3. AI Models and Technical Implementation

#### Computer Vision Models
- The current implementation uses **OpenCV's built-in algorithms** for player detection and tracking:
  - **Background Subtraction (MOG2)**: To detect moving objects against the field background
  - **Contour Detection**: To identify player silhouettes
  - **HSV Color Analysis**: For team jersey differentiation
  - **Proximity-Based Tracking**: To maintain player identity across frames

#### Local Large Language Models
- The system is designed to utilize various open-source LLMs for analysis:
  - **Llama 2 (7B)**: Meta's powerful open-source LLM for general analysis
  - **Mistral 7B**: High-performance model with strong reasoning capabilities
  - **Phi-2**: Microsoft's compact but capable model for systems with less GPU memory
  - **GPT-J 6B**: EleutherAI's generation-focused model

#### Pass Detection Algorithm
The pass detection process works through these specific steps:
1. **Ball Position Tracking**: Records ball coordinates through sequential frames
2. **Velocity and Direction Analysis**: Calculates the velocity vector and detects significant changes in direction or speed
3. **Angle Change Detection**: Measures the angle between consecutive ball movement vectors (passes often involve >30° angle changes)
4. **Player Proximity Analysis**: Identifies which players are closest to the ball before and after a potential pass
5. **Team Verification**: Confirms that the potential passer and receiver are on the same team
6. **Distance Measurement**: Calculates the pass distance to classify it as short, medium, or long

In the current implementation, a pass is detected when:
- The ball changes direction by more than 30 degrees, OR
- The ball speed changes by more than 50% from one frame to the next
- Two players from the same team are identified as the closest players to the ball before and after the direction/speed change

## System Requirements

- **OS**: Windows 10/11, macOS, or Linux
- **CPU**: Quad-core processor (Intel i5/i7 or AMD Ryzen 5/7)
- **RAM**: Minimum 16GB (32GB recommended)
- **GPU**: NVIDIA GPU with at least 8GB VRAM (for optimal performance)
- **Storage**: 10GB for the application, plus space for video files
- **Python**: 3.10+

## Installation

### Using Virtual Environment

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/Gaelic-Football-Match-Analyser.git
   cd Gaelic-Football-Match-Analyser
   ```

2. Create and activate a Python virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Processing Videos

To analyze a video file:

```
python main.py --video_path data/videos/your_video.mp4 --output_dir data/output
```

For just video processing without model loading (faster):

```
python -c "from src.data_processing.video_processor import VideoProcessor; processor = VideoProcessor(); processor.process('data/videos/your_video.mp4', 'data/output')"
```

To specifically detect passes:

```
python -c "from src.data_processing.video_processor import VideoProcessor; processor = VideoProcessor(); passes = processor.detect_passes('data/videos/your_video.mp4', 'data/output'); print(f'Detected {len(passes)} passes')"
```

### Options and Parameters

- `--video_path`: Path to the input video file
- `--output_dir`: Directory to save output files (defaults to "output")
- `--model`: LLM to use for analysis (options: llama-2-7b, mistral-7b, phi-2, gpt-j-6b)
- `--skip_video_processing`: Skip video processing if already done
- `--skip_fine_tuning`: Skip model fine-tuning step

## Project Structure

```
Gaelic-Football-Match-Analyser/
├── requirements.txt      # Python dependencies
├── main.py               # Main entry point
├── data/                 # Data directory
│   ├── videos/           # Input video files
│   ├── output/           # Analysis output
│   └── processed/        # Processed data
├── src/                  # Source code
│   ├── data_processing/  # Video processing and data transformation
│   ├── model/            # LLM selection and fine-tuning
│   ├── inference/        # Inference pipeline and analysis
│   └── ui/               # User interface
```

## Future Improvements

- Replace simulated ball detection with a dedicated object detection model
- Implement more sophisticated player tracking algorithms
- Add support for tactical formation recognition
- Enhance the UI with more interactive visualizations

## Security Best Practices

When using this project, please follow these security practices:

1. **API Keys and Credentials**: 
   - Never commit API keys or credentials to GitHub
   - Use the provided `config.template.json` to create your own `config.json` file (which is gitignored)
   - Store sensitive credentials in environment variables when possible

2. **Video Data**:
   - Be mindful of privacy concerns when working with match footage
   - The `data/videos/`, `data/output/`, and `data/processed/` directories are gitignored to prevent accidentally committing large video files

3. **Model Files**:
   - Large model files (`.bin`, `.pt`, `.pth`, `.onnx`, `.safetensors`) are gitignored
   - Download models directly from their respective sources or using the application

4. **Before Committing**:
   - Run `git status` to verify which files will be committed
   - Check `.gitignore` if you need to add additional patterns for sensitive files

## License

[MIT License](LICENSE)

## Acknowledgements

- The open-source LLM community
- Contributors to the PyTorch, Transformers, and OpenCV libraries
- Sports analysts and coaches who provided domain expertise
