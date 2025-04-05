"""
Video Processing Module for Gaelic Football Match Analyser

This module handles:
1. Video loading and frame extraction
2. Player detection and tracking
3. Field segmentation
4. Play segmentation (identifying key moments)
5. Export of processed video data
"""

import cv2
import numpy as np
import logging
import os
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)

class VideoProcessor:
    """
    Process football/Gaelic football videos to extract relevant information
    for further analysis.
    """
    
    def __init__(self, 
                 detection_interval: int = 30,
                 confidence_threshold: float = 0.6,
                 track_max_age: int = 20,
                 player_detection_model: str = "yolov8"):
        """
        Initialize the video processor.
        
        Args:
            detection_interval: Number of frames between detection runs
            confidence_threshold: Confidence threshold for detections
            track_max_age: Maximum age of a track before it's deleted
            player_detection_model: Model to use for player detection
        """
        self.detection_interval = detection_interval
        self.confidence_threshold = confidence_threshold
        self.track_max_age = track_max_age
        self.player_detection_model = player_detection_model
        
        # Initialize models based on availability
        self.initialize_models()
        
    def initialize_models(self):
        """Initialize detection and tracking models."""
        logger.info("Initializing detection and tracking models")
        
        # For simplicity, we're using OpenCV's built-in detectors and trackers
        # In a production system, you'd use more advanced models like YOLOv8
        
        try:
            # Initialize player detector (pretend this is a proper YOLO model)
            self.player_detector = cv2.createBackgroundSubtractorMOG2()
            logger.info("Using BackgroundSubtractorMOG2 for player detection")
            
            # Initialize tracker - handle both older and newer OpenCV versions
            try:
                # Try to use legacy tracker if available
                self.tracker = cv2.legacy.TrackerCSRT_create()
                logger.info("Using CSRT tracker for player tracking")
            except AttributeError:
                # Fallback for newer OpenCV versions without legacy module
                logger.info("Legacy trackers not available, using simple tracking implementation")
                self.tracker = None  # We'll use our custom tracking in _update_tracks
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
            
    def process(self, video_path: str, output_dir: str = "output") -> Dict[str, Any]:
        """
        Process a video file and extract relevant information.
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save output files
            
        Returns:
            Dict containing processed data
        """
        logger.info(f"Processing video: {video_path}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Open video
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video properties: {width}x{height}, {fps} fps, {frame_count} frames")
        
        # Output data structure
        output_data = {
            "video_metadata": {
                "filename": os.path.basename(video_path),
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "processed_date": datetime.now().isoformat()
            },
            "frames": [],
            "plays": [],
            "player_tracks": [],
            "field_data": self._detect_field(video)
        }
        
        # Process frames
        frame_idx = 0
        plays = []
        current_play = None
        player_tracks = {}
        
        while True:
            ret, frame = video.read()
            if not ret:
                break
                
            # Process every N frames for efficiency
            if frame_idx % self.detection_interval == 0:
                logger.debug(f"Processing frame {frame_idx}/{frame_count}")
                
                # Detect players
                player_positions = self._detect_players(frame)
                
                # Update tracks
                player_tracks = self._update_tracks(player_tracks, player_positions, frame_idx)
                
                # Check if this is a key moment (new play)
                is_key_frame, play_type = self._detect_key_moment(frame, player_positions)
                
                if is_key_frame:
                    if current_play:
                        current_play["end_frame"] = frame_idx
                        plays.append(current_play)
                    
                    current_play = {
                        "start_frame": frame_idx,
                        "end_frame": None,
                        "play_type": play_type,
                        "key_players": []
                    }
                
                # Save sample frames (not all frames to save space)
                if frame_idx % (self.detection_interval * 10) == 0:
                    frame_filename = f"frame_{frame_idx:06d}.jpg"
                    frame_path = os.path.join(output_dir, frame_filename)
                    cv2.imwrite(frame_path, frame)
                    
                    output_data["frames"].append({
                        "frame_idx": frame_idx,
                        "filename": frame_filename,
                        "player_positions": player_positions,
                        "is_key_frame": is_key_frame,
                        "play_type": play_type if is_key_frame else None
                    })
            
            frame_idx += 1
        
        # Save last play if exists
        if current_play:
            current_play["end_frame"] = frame_idx - 1
            plays.append(current_play)
        
        # Add plays to output data
        output_data["plays"] = plays
        
        # Convert player tracks to list for JSON serialization
        output_data["player_tracks"] = [
            {"player_id": player_id, "positions": positions}
            for player_id, positions in player_tracks.items()
        ]
        
        # Save metadata
        metadata_path = os.path.join(output_dir, "video_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Processed {frame_idx} frames, detected {len(plays)} plays")
        logger.info(f"Output saved to {metadata_path}")
        
        video.release()
        return output_data
    
    def _detect_players(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect players in a frame and identify their team based on jersey color.
        
        Args:
            frame: Input video frame
            
        Returns:
            List of player positions, team identification, and bounding boxes
        """
        # Apply background subtraction to find moving objects
        fgmask = self.player_detector.apply(frame)
        
        # Find contours of moving objects
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        player_positions = []
        
        for i, contour in enumerate(contours):
            # Filter small contours
            if cv2.contourArea(contour) < 500:
                continue
                
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Simple heuristic to filter out non-players (too large or small)
            if w * h > 50000 or w * h < 1000 or h < w:
                continue
            
            # Extract region for team identification
            player_roi = frame[y:y+h, x:x+w]
            
            # Team identification based on dominant color
            team = self._identify_team(player_roi)
                
            # Center point of the player
            center_x = x + w // 2
            center_y = y + h // 2
            
            player_positions.append({
                "id": i,
                "bbox": [x, y, w, h],
                "center": [center_x, center_y],
                "team": team,
                "confidence": 0.8  # Placeholder, would be from actual detector
            })
        
        return player_positions
    
    def _identify_team(self, player_roi: np.ndarray) -> str:
        """
        Identify which team a player belongs to based on jersey color.
        
        Args:
            player_roi: Region of interest containing the player
            
        Returns:
            Team identifier ("team_a", "team_b", or "unknown")
        """
        try:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(player_roi, cv2.COLOR_BGR2HSV)
            
            # Define color ranges for teams
            # These should be calibrated for actual jersey colors
            team_a_lower = np.array([100, 50, 50])  # Blue-ish
            team_a_upper = np.array([140, 255, 255])
            
            team_b_lower = np.array([0, 50, 50])    # Red-ish
            team_b_upper = np.array([20, 255, 255])
            
            # Create masks for each team
            mask_a = cv2.inRange(hsv, team_a_lower, team_a_upper)
            mask_b = cv2.inRange(hsv, team_b_lower, team_b_upper)
            
            # Count pixels for each team
            team_a_pixels = cv2.countNonZero(mask_a)
            team_b_pixels = cv2.countNonZero(mask_b)
            
            # Total ROI pixels
            total_pixels = player_roi.shape[0] * player_roi.shape[1]
            
            # Threshold for team assignment (at least 10% of pixels match team color)
            threshold = total_pixels * 0.1
            
            if team_a_pixels > threshold and team_a_pixels > team_b_pixels:
                return "team_a"
            elif team_b_pixels > threshold:
                return "team_b"
            else:
                return "unknown"
        except Exception:
            # In case of issues with color processing
            return "unknown"
    
    def _update_tracks(self, 
                      tracks: Dict[int, List], 
                      player_positions: List[Dict], 
                      frame_idx: int) -> Dict[int, List]:
        """
        Update player tracks with new detections.
        
        Args:
            tracks: Existing player tracks
            player_positions: New player positions
            frame_idx: Current frame index
            
        Returns:
            Updated tracks
        """
        # Simple tracking by proximity (in real implementation, use a proper tracker)
        new_tracks = tracks.copy()
        
        # If no existing tracks, create new ones
        if not tracks:
            for player in player_positions:
                new_tracks[player["id"]] = [{
                    "frame": frame_idx,
                    "position": player["center"],
                    "bbox": player["bbox"],
                    "team": player.get("team", "unknown")
                }]
            return new_tracks
        
        # Match new detections to existing tracks
        for player in player_positions:
            matched = False
            min_dist = float('inf')
            min_id = None
            
            for track_id, track_data in tracks.items():
                if not track_data:
                    continue
                    
                last_pos = track_data[-1]["position"]
                dx = player["center"][0] - last_pos[0]
                dy = player["center"][1] - last_pos[1]
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist < min_dist:
                    min_dist = dist
                    min_id = track_id
            
            # If close enough, update existing track
            if min_dist < 50:  # Threshold for matching
                if min_id in new_tracks:
                    new_tracks[min_id].append({
                        "frame": frame_idx,
                        "position": player["center"],
                        "bbox": player["bbox"],
                        "team": player.get("team", "unknown")
                    })
                matched = True
            
            # Otherwise, create new track
            if not matched:
                new_id = max(tracks.keys()) + 1 if tracks else player["id"]
                new_tracks[new_id] = [{
                    "frame": frame_idx,
                    "position": player["center"],
                    "bbox": player["bbox"],
                    "team": player.get("team", "unknown")
                }]
        
        return new_tracks
    
    def _detect_field(self, video: cv2.VideoCapture) -> Dict[str, Any]:
        """
        Detect the field boundaries and markings.
        
        Args:
            video: Video capture object
            
        Returns:
            Dict containing field data
        """
        # This is a simplified placeholder that would be replaced with
        # actual field detection using computer vision techniques
        
        # Reset video to beginning
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Read a frame
        ret, frame = video.read()
        if not ret:
            return {}
        
        height, width = frame.shape[:2]
        
        # Simple placeholder field data - in a real system this would be detected
        field_data = {
            "field_type": "gaelic_football",
            "boundaries": {
                "top_left": [0.05 * width, 0.05 * height],
                "top_right": [0.95 * width, 0.05 * height],
                "bottom_left": [0.05 * width, 0.95 * height],
                "bottom_right": [0.95 * width, 0.95 * height]
            },
            "goal_positions": {
                "left": [0.05 * width, 0.5 * height],
                "right": [0.95 * width, 0.5 * height]
            },
            "field_markings": []
        }
        
        return field_data
    
    def _detect_key_moment(self, 
                          frame: np.ndarray, 
                          player_positions: List[Dict]) -> Tuple[bool, Optional[str]]:
        """
        Detect if current frame represents a key moment (start of a new play).
        
        Args:
            frame: Current video frame
            player_positions: Detected player positions
            
        Returns:
            Tuple of (is_key_moment, play_type)
        """
        # This is a simplified placeholder that would be replaced with
        # actual key moment detection using player positions and motion analysis
        
        # In a real implementation, we would:
        # 1. Look for specific formations (e.g., players gathered for a kick-off)
        # 2. Detect sudden changes in player velocities
        # 3. Recognize referee signals
        # 4. Use ball detection to identify possessions
        
        # For demonstration, randomly identify ~5% of frames as key moments
        if np.random.random() < 0.01:  # Very low probability for demonstration
            play_types = ["kickout", "free_kick", "sideline_kick", "penalty", "open_play"]
            return True, np.random.choice(play_types)
        
        return False, None

    # Add new method for pass detection
    def detect_passes(self, video_path: str, output_dir: str = "output") -> List[Dict[str, Any]]:
        """
        Detect passes between players in a video.
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save output files
            
        Returns:
            List of detected passes
        """
        logger.info(f"Detecting passes in video: {video_path}")
        
        # Process video to get player tracks
        processed_data = self.process(video_path, output_dir)
        
        # Extract player tracks
        player_tracks = {}
        for track in processed_data["player_tracks"]:
            player_id = track["player_id"]
            player_tracks[player_id] = track["positions"]
        
        # Detect ball (simplified - in a real implementation, this would be more sophisticated)
        ball_positions = self._detect_ball_positions(video_path)
        
        # Analyze ball movement to detect passes
        passes = self._analyze_ball_for_passes(ball_positions, player_tracks)
        
        # Save pass data
        passes_path = os.path.join(output_dir, "detected_passes.json")
        with open(passes_path, 'w') as f:
            json.dump(passes, f, indent=2)
        
        logger.info(f"Detected {len(passes)} passes, saved to {passes_path}")
        
        return passes
    
    def _detect_ball_positions(self, video_path: str) -> List[Dict[str, Any]]:
        """
        Detect ball positions throughout the video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of ball positions per frame
        """
        # This is a simplified placeholder that would be replaced with
        # actual ball detection using computer vision techniques
        
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        ball_positions = []
        frame_idx = 0
        
        # Parameters for ball detection
        ball_lower = np.array([0, 0, 200])  # Very bright objects (simplified)
        ball_upper = np.array([180, 30, 255])
        
        # Random starting position for simulation
        ball_x = 0.5  # Normalized coordinates (0-1)
        ball_y = 0.5
        
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        while True:
            ret, frame = video.read()
            if not ret:
                break
                
            # Process every N frames for efficiency
            if frame_idx % self.detection_interval == 0:
                # In a real implementation, this would use dedicated ball detection
                # For simplicity, we'll simulate ball movement
                
                # Simulate ball movement (in a real implementation, this would be detected)
                if np.random.random() < 0.1:  # Occasionally change direction
                    ball_x += np.random.uniform(-0.1, 0.1)
                    ball_y += np.random.uniform(-0.1, 0.1)
                    
                    # Keep ball in bounds
                    ball_x = max(0, min(1, ball_x))
                    ball_y = max(0, min(1, ball_y))
                
                # Add noise to simulate imperfect detection
                noisy_x = ball_x + np.random.normal(0, 0.01)
                noisy_y = ball_y + np.random.normal(0, 0.01)
                
                # Convert to pixel coordinates
                pixel_x = int(noisy_x * width)
                pixel_y = int(noisy_y * height)
                
                # Store ball position
                ball_positions.append({
                    "frame": frame_idx,
                    "position": [pixel_x, pixel_y],
                    "confidence": np.random.uniform(0.6, 0.95)
                })
            
            frame_idx += 1
        
        video.release()
        return ball_positions
    
    def _analyze_ball_for_passes(self, 
                               ball_positions: List[Dict[str, Any]], 
                               player_tracks: Dict[int, List]) -> List[Dict[str, Any]]:
        """
        Analyze ball movement to detect passes between players.
        
        Args:
            ball_positions: List of ball positions per frame
            player_tracks: Dict of player tracks
            
        Returns:
            List of detected passes
        """
        passes = []
        
        # Need at least a few ball positions to detect passes
        if len(ball_positions) < 5:
            return passes
        
        # Analyze ball movement
        for i in range(1, len(ball_positions) - 1):
            prev_pos = ball_positions[i-1]["position"]
            curr_pos = ball_positions[i]["position"]
            next_pos = ball_positions[i+1]["position"]
            
            # Look for significant changes in ball direction or speed
            # (simplified - real implementation would be more sophisticated)
            dx1 = curr_pos[0] - prev_pos[0]
            dy1 = curr_pos[1] - prev_pos[1]
            dx2 = next_pos[0] - curr_pos[0]
            dy2 = next_pos[1] - curr_pos[1]
            
            # Calculate velocities
            v1 = np.sqrt(dx1*dx1 + dy1*dy1)
            v2 = np.sqrt(dx2*dx2 + dy2*dy2)
            
            # Calculate angle change
            if v1 > 0 and v2 > 0:
                cos_angle = (dx1*dx2 + dy1*dy2) / (v1 * v2)
                cos_angle = max(-1, min(1, cos_angle))  # Clamp to [-1, 1]
                angle_change = np.arccos(cos_angle) * 180 / np.pi
            else:
                angle_change = 0
            
            # Check for potential pass (significant direction change or speed change)
            is_pass = (angle_change > 30) or (abs(v2 - v1) > v1 * 0.5)
            
            if is_pass:
                # Find closest players to the ball before and after the potential pass
                frame = ball_positions[i]["frame"]
                passer = self._find_closest_player(prev_pos, player_tracks, frame - self.detection_interval)
                receiver = self._find_closest_player(next_pos, player_tracks, frame + self.detection_interval)
                
                # Verify this is a valid pass (different players, close enough)
                if (passer and receiver and passer["player_id"] != receiver["player_id"] and
                    passer.get("team") == receiver.get("team")):
                    
                    # Calculate pass info
                    pass_distance = np.sqrt(
                        (receiver["position"][0] - passer["position"][0])**2 +
                        (receiver["position"][1] - passer["position"][1])**2
                    )
                    
                    # Add to passes list
                    passes.append({
                        "frame": frame,
                        "passer_id": passer["player_id"],
                        "receiver_id": receiver["player_id"],
                        "passer_position": passer["position"],
                        "receiver_position": receiver["position"],
                        "team": passer.get("team", "unknown"),
                        "distance": float(pass_distance),
                        "confidence": 0.7,  # Placeholder confidence
                        "pass_type": self._determine_pass_type(pass_distance)
                    })
        
        return passes
    
    def _find_closest_player(self, 
                           ball_position: List[float], 
                           player_tracks: Dict[int, List],
                           frame: int) -> Optional[Dict[str, Any]]:
        """
        Find the player closest to a given position in a specific frame.
        
        Args:
            ball_position: [x, y] position to find closest player to
            player_tracks: Dict of player tracks
            frame: Frame number to check
            
        Returns:
            Dict with closest player info or None if no player is close enough
        """
        min_dist = float('inf')
        closest_player = None
        
        for player_id, positions in player_tracks.items():
            # Find the position record closest to the requested frame
            closest_record = min(positions, key=lambda p: abs(p["frame"] - frame)) if positions else None
            
            if not closest_record:
                continue
            
            # Only consider records reasonably close to the requested frame
            if abs(closest_record["frame"] - frame) > self.detection_interval * 2:
                continue
            
            # Calculate distance to ball
            player_pos = closest_record["position"]
            dx = player_pos[0] - ball_position[0]
            dy = player_pos[1] - ball_position[1]
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist < min_dist:
                min_dist = dist
                closest_player = {
                    "player_id": player_id,
                    "position": player_pos,
                    "team": closest_record.get("team", "unknown"),
                    "distance": dist
                }
        
        # Only return if player is reasonably close to the ball (threshold could be tuned)
        return closest_player if closest_player and closest_player["distance"] < 100 else None
    
    def _determine_pass_type(self, distance: float) -> str:
        """
        Determine the type of pass based on distance.
        
        Args:
            distance: Pass distance in pixels
            
        Returns:
            Pass type description
        """
        if distance < 100:
            return "short_pass"
        elif distance < 300:
            return "medium_pass"
        else:
            return "long_pass"
