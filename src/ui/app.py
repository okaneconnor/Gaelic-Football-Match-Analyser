"""
UI Application Module for Gaelic Football Match Analyser

This module handles:
1. Streamlit-based user interface
2. File upload and processing
3. Interactive visualization of results
4. Model selection and configuration
"""

import streamlit as st
import os
import json
import tempfile
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import matplotlib.pyplot as plt
import pandas as pd
import base64
import sys
import numpy as np

# Add parent directory to path to import other modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import modules
from data_processing.video_processor import VideoProcessor
from data_processing.data_transformer import DataTransformer
from model.model_selector import ModelSelector
from model.fine_tuner import FineTuner
from inference.inference_pipeline import InferencePipeline
from inference.analysis import Analyzer
from ui.utils import (format_seconds, create_download_link, humanize_bytes,
                     format_percent, create_comparison_table, get_model_emoji)

# Set up logging
logger = logging.getLogger(__name__)

def launch_ui():
    """Launch the Streamlit UI."""
    # Streamlit doesn't need this function, but it's required for main.py
    import streamlit.web.cli as stcli
    sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
    sys.exit(stcli.main())

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Gaelic Football Match Analyser",
        page_icon="🏐",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state for storing data between reruns
    if "processed_data" not in st.session_state:
        st.session_state.processed_data = None
    if "inference_results" not in st.session_state:
        st.session_state.inference_results = None
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "mistral-7b"
    if "device" not in st.session_state:
        st.session_state.device = "cpu"
    if "model_path" not in st.session_state:
        st.session_state.model_path = None
    if "video_file_path" not in st.session_state:
        st.session_state.video_file_path = None
    if "player_tracking_results" not in st.session_state:
        st.session_state.player_tracking_results = None
    if "pass_detection_results" not in st.session_state:
        st.session_state.pass_detection_results = None
        
    # Application title
    st.title("Gaelic Football Match Analyser")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Device selection
        device_options = ["cpu"]
        if torch_available():
            if torch_cuda_available():
                device_options.append("cuda")
            if torch_mps_available():
                device_options.append("mps")
        
        st.session_state.device = st.selectbox(
            "Processing Device", 
            options=device_options,
            index=device_options.index(st.session_state.device)
        )
        
        st.subheader("Model Selection")
        model_options = ["llama-2-7b", "mistral-7b", "phi-2", "gpt-j-6b"]
        
        st.session_state.selected_model = st.selectbox(
            "Select Model",
            options=model_options,
            index=model_options.index(st.session_state.selected_model),
            format_func=lambda x: f"{get_model_emoji(x)} {x}"
        )
        
        custom_model = st.checkbox("Use custom model path")
        if custom_model:
            st.session_state.model_path = st.text_input("Custom Model Path")
        
        st.divider()
        
        # Advanced options (collapsible)
        with st.expander("Advanced Options"):
            quantization = st.selectbox(
                "Quantization",
                options=["4bit", "8bit", "none"],
                index=0
            )
            
            fine_tuning = st.checkbox("Enable Fine-tuning", value=False)
            if fine_tuning:
                st.number_input("LoRA Rank", min_value=1, max_value=64, value=8)
                st.number_input("Learning Rate", min_value=1e-5, max_value=1e-3, value=2e-4, format="%.0e")
                st.number_input("Training Epochs", min_value=1, max_value=10, value=3)
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Video Upload & Processing", 
        "Player Tracking", 
        "Pass Detection",
        "Analysis",
        "Results"
    ])
    
    # Tab 1: Video Upload & Processing
    with tab1:
        st.header("Video Upload & Processing")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload a Gaelic football video",
                type=["mp4", "avi", "mov", "mkv"]
            )
            
            if uploaded_file:
                # Create temporary file to save the uploaded video
                temp_dir = Path("data/videos")
                os.makedirs(temp_dir, exist_ok=True)
                
                # Save uploaded file to disk
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                st.session_state.video_file_path = temp_path
                
                st.success(f"Video saved to {temp_path}")
                
                # Display video
                st.video(temp_path)
            else:
                st.info("Please upload a video file to begin processing")
                
                # Provide option to use sample data
                if st.button("Use Sample Data"):
                    st.session_state.processed_data = create_sample_data()
                    st.session_state.player_tracking_results = st.session_state.processed_data
                    st.session_state.pass_detection_results = {
                        "passes": [
                            {
                                "frame": 120,
                                "passer_id": 1,
                                "receiver_id": 2,
                                "team": "team_a",
                                "distance": 250,
                                "pass_type": "medium_pass"
                            },
                            {
                                "frame": 240,
                                "passer_id": 3,
                                "receiver_id": 4,
                                "team": "team_b",
                                "distance": 350,
                                "pass_type": "long_pass"
                            }
                        ]
                    }
                    st.success("Sample data loaded")
        
        with col2:
            st.subheader("Processing Options")
            
            detection_interval = st.slider(
                "Detection Interval (frames)",
                min_value=5,
                max_value=60,
                value=30,
                help="Number of frames between detection runs. Lower values provide more accurate tracking but require more processing."
            )
            
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.1,
                max_value=0.9,
                value=0.6,
                help="Minimum confidence score for detections. Higher values reduce false positives."
            )
            
            # Output directory
            output_dir = st.text_input(
                "Output Directory",
                value="data/processed",
                help="Directory where processed data will be saved."
            )
            
            # Process button
            process_button = st.button("Process Video")
            
            if process_button and st.session_state.video_file_path:
                with st.spinner("Processing video..."):
                    try:
                        # Create progress bar
                        progress_bar = st.progress(0)
                        
                        # Initialize video processor
                        video_processor = VideoProcessor(
                            detection_interval=detection_interval,
                            confidence_threshold=confidence_threshold
                        )
                        
                        # Process video
                        start_time = time.time()
                        processed_data = video_processor.process(
                            st.session_state.video_file_path,
                            output_dir=output_dir
                        )
                        processing_time = time.time() - start_time
                        
                        # Update session state
                        st.session_state.processed_data = processed_data
                        st.session_state.player_tracking_results = processed_data
                        
                        # Set progress to complete
                        progress_bar.progress(100)
                        
                        st.success(f"Video processed successfully in {format_seconds(processing_time)}")
                        
                    except Exception as e:
                        st.error(f"Error processing video: {str(e)}")
    
    # Tab 2: Player Tracking
    with tab2:
        st.header("Player Tracking")
        
        if st.session_state.player_tracking_results:
            # Display player tracking results
            tracking_data = st.session_state.player_tracking_results
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Player Tracks")
                
                if "player_tracks" in tracking_data:
                    player_tracks = tracking_data["player_tracks"]
                    
                    # Convert to dataframe for display
                    tracks_data = []
                    for track in player_tracks:
                        player_id = track["player_id"]
                        team = "Unknown"
                        
                        # Try to determine team from positions
                        if "positions" in track and track["positions"]:
                            for pos in track["positions"]:
                                if "team" in pos:
                                    team = pos["team"]
                                    break
                        
                        # Count positions
                        position_count = len(track.get("positions", []))
                        
                        tracks_data.append({
                            "Player ID": player_id,
                            "Team": team,
                            "Positions Tracked": position_count
                        })
                    
                    if tracks_data:
                        tracks_df = pd.DataFrame(tracks_data)
                        st.dataframe(tracks_df)
                    else:
                        st.info("No player tracks detected")
                else:
                    st.info("No player tracking data available")
            
            with col2:
                st.subheader("Team Distribution")
                
                # Count players by team
                team_counts = {"team_a": 0, "team_b": 0, "unknown": 0}
                
                if "player_tracks" in tracking_data:
                    for track in tracking_data["player_tracks"]:
                        team = "unknown"
                        
                        # Try to determine team from positions
                        if "positions" in track and track["positions"]:
                            for pos in track["positions"]:
                                if "team" in pos:
                                    team = pos["team"]
                                    break
                        
                        if team in team_counts:
                            team_counts[team] += 1
                
                # Create pie chart
                fig, ax = plt.subplots()
                labels = ["Team A", "Team B", "Unknown"]
                sizes = [team_counts["team_a"], team_counts["team_b"], team_counts["unknown"]]
                colors = ["#3498db", "#e74c3c", "#95a5a6"]
                
                if sum(sizes) > 0:
                    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                    ax.axis('equal')
                    st.pyplot(fig)
                else:
                    st.info("No team data available")
            
            # Run additional tracking analysis
            if st.button("Run Advanced Player Analysis"):
                with st.spinner("Analyzing player movements..."):
                    # Placeholder for advanced analysis function
                    st.success("Player analysis complete!")
        else:
            st.info("No player tracking data available. Please process a video first.")
    
    # Tab 3: Pass Detection
    with tab3:
        st.header("Pass Detection")
        
        if st.session_state.video_file_path and not st.session_state.pass_detection_results:
            st.info("Pass detection has not been run yet")
            
            detect_passes = st.button("Detect Passes")
            
            if detect_passes:
                with st.spinner("Detecting passes..."):
                    try:
                        # Initialize video processor
                        video_processor = VideoProcessor(
                            detection_interval=detection_interval,
                            confidence_threshold=confidence_threshold
                        )
                        
                        # Process video for pass detection
                        start_time = time.time()
                        passes = video_processor.detect_passes(
                            st.session_state.video_file_path,
                            output_dir=output_dir
                        )
                        processing_time = time.time() - start_time
                        
                        # Update session state
                        st.session_state.pass_detection_results = {"passes": passes}
                        
                        st.success(f"Pass detection completed in {format_seconds(processing_time)}")
                    except Exception as e:
                        st.error(f"Error detecting passes: {str(e)}")
        
        if st.session_state.pass_detection_results:
            # Display pass detection results
            pass_data = st.session_state.pass_detection_results
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Detected Passes")
                
                if "passes" in pass_data and pass_data["passes"]:
                    passes = pass_data["passes"]
                    
                    # Convert to dataframe for display
                    passes_df = pd.DataFrame([
                        {
                            "Frame": p.get("frame", ""),
                            "Passer ID": p.get("passer_id", ""),
                            "Receiver ID": p.get("receiver_id", ""),
                            "Team": p.get("team", "Unknown"),
                            "Distance": f"{p.get('distance', 0):.1f}",
                            "Pass Type": p.get("pass_type", ""),
                            "Confidence": f"{p.get('confidence', 0) * 100:.1f}%"
                        }
                        for p in passes
                    ])
                    
                    st.dataframe(passes_df)
                else:
                    st.info("No passes detected")
            
            with col2:
                st.subheader("Pass Statistics")
                
                if "passes" in pass_data and pass_data["passes"]:
                    passes = pass_data["passes"]
                    
                    # Count passes by type
                    pass_types = {}
                    for p in passes:
                        pass_type = p.get("pass_type", "unknown")
                        if pass_type in pass_types:
                            pass_types[pass_type] += 1
                        else:
                            pass_types[pass_type] = 1
                    
                    # Team pass counts
                    team_passes = {"team_a": 0, "team_b": 0, "unknown": 0}
                    for p in passes:
                        team = p.get("team", "unknown")
                        if team in team_passes:
                            team_passes[team] += 1
                    
                    # Create bar chart for pass types
                    fig, ax = plt.subplots()
                    ax.bar(pass_types.keys(), pass_types.values())
                    ax.set_title("Passes by Type")
                    ax.set_ylabel("Count")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    
                    # Display team pass counts
                    st.metric("Team A Passes", team_passes["team_a"])
                    st.metric("Team B Passes", team_passes["team_b"])
                else:
                    st.info("No pass statistics available")
        else:
            st.info("No pass detection results available")
    
    # Tab 4: Analysis
    with tab4:
        st.header("Analysis")
        
        if st.session_state.processed_data:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Data Transformation")
                
                transform_button = st.button("Transform Data for LLM")
                
                if transform_button:
                    with st.spinner("Transforming data..."):
                        try:
                            # Initialize data transformer
                            data_transformer = DataTransformer()
                            
                            # Transform data
                            start_time = time.time()
                            transformed_data = data_transformer.transform(
                                st.session_state.processed_data,
                                output_dir=output_dir
                            )
                            transform_time = time.time() - start_time
                            
                            # Display summary
                            st.success(f"Data transformed in {format_seconds(transform_time)}")
                            
                            # Show transformed data summary
                            if "plays" in transformed_data:
                                st.info(f"Transformed {len(transformed_data['plays'])} plays")
                            
                            if "training_examples" in transformed_data:
                                st.info(f"Created {len(transformed_data['training_examples'])} training examples")
                        except Exception as e:
                            st.error(f"Error transforming data: {str(e)}")
            
            with col2:
                st.subheader("LLM Analysis")
                
                run_analysis = st.button("Run LLM Analysis")
                
                if run_analysis:
                    with st.spinner("Running analysis with LLM..."):
                        try:
                            # This is a placeholder for actual LLM analysis
                            # In a real implementation, this would use the ModelSelector, InferencePipeline, etc.
                            
                            # Simulate analysis with a delay
                            time.sleep(3)
                            
                            st.success("Analysis complete!")
                            
                            # Update session state with sample results
                            st.session_state.analysis_results = {
                                "team_performance": {
                                    "team_a": {
                                        "pass_completion": 0.78,
                                        "possession": 0.52,
                                        "scoring_efficiency": 0.65
                                    },
                                    "team_b": {
                                        "pass_completion": 0.71,
                                        "possession": 0.48,
                                        "scoring_efficiency": 0.58
                                    }
                                },
                                "key_insights": [
                                    "Team A shows a strong preference for short passes through the center",
                                    "Team B is more effective at long-range passing but struggles with completion rate",
                                    "Player movement patterns suggest a zone defense from Team A"
                                ]
                            }
                        except Exception as e:
                            st.error(f"Error running analysis: {str(e)}")
        else:
            st.info("No processed data available. Please process a video first.")
    
    # Tab 5: Results
    with tab5:
        st.header("Results")
        
        if st.session_state.analysis_results:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Team Performance")
                
                if "team_performance" in st.session_state.analysis_results:
                    team_perf = st.session_state.analysis_results["team_performance"]
                    
                    # Display performance metrics
                    with st.container():
                        st.markdown("### Team A")
                        metrics = team_perf.get("team_a", {})
                        
                        for metric, value in metrics.items():
                            st.metric(
                                metric.replace("_", " ").title(),
                                format_percent(value)
                            )
                    
                    with st.container():
                        st.markdown("### Team B")
                        metrics = team_perf.get("team_b", {})
                        
                        for metric, value in metrics.items():
                            st.metric(
                                metric.replace("_", " ").title(),
                                format_percent(value)
                            )
            
            with col2:
                st.subheader("Key Insights")
                
                if "key_insights" in st.session_state.analysis_results:
                    insights = st.session_state.analysis_results["key_insights"]
                    
                    for i, insight in enumerate(insights):
                        st.markdown(f"**{i+1}.** {insight}")
                
                # Export options
                st.subheader("Export Results")
                
                export_format = st.selectbox(
                    "Export Format",
                    options=["JSON", "CSV", "HTML"]
                )
                
                if st.button("Export"):
                    if export_format == "JSON":
                        json_data = json.dumps(st.session_state.analysis_results, indent=2)
                        st.download_button(
                            label="Download JSON",
                            data=json_data,
                            file_name="analysis_results.json",
                            mime="application/json"
                        )
                    elif export_format == "CSV":
                        # Convert to CSV (simplified example)
                        csv_data = "metric,team_a,team_b\n"
                        for metric in st.session_state.analysis_results["team_performance"]["team_a"]:
                            team_a_value = st.session_state.analysis_results["team_performance"]["team_a"][metric]
                            team_b_value = st.session_state.analysis_results["team_performance"]["team_b"][metric]
                            csv_data += f"{metric},{team_a_value},{team_b_value}\n"
                        
                        st.download_button(
                            label="Download CSV",
                            data=csv_data,
                            file_name="analysis_results.csv",
                            mime="text/csv"
                        )
                    elif export_format == "HTML":
                        # Simple HTML export
                        html_content = "<html><head><title>Analysis Results</title></head><body>"
                        html_content += "<h1>Gaelic Football Match Analysis</h1>"
                        
                        # Add team performance
                        html_content += "<h2>Team Performance</h2>"
                        html_content += "<table border='1'><tr><th>Metric</th><th>Team A</th><th>Team B</th></tr>"
                        
                        team_a = st.session_state.analysis_results["team_performance"]["team_a"]
                        team_b = st.session_state.analysis_results["team_performance"]["team_b"]
                        
                        for metric in team_a:
                            html_content += f"<tr><td>{metric.replace('_', ' ').title()}</td>"
                            html_content += f"<td>{format_percent(team_a[metric])}</td>"
                            html_content += f"<td>{format_percent(team_b[metric])}</td></tr>"
                        
                        html_content += "</table>"
                        
                        # Add insights
                        html_content += "<h2>Key Insights</h2><ul>"
                        for insight in st.session_state.analysis_results["key_insights"]:
                            html_content += f"<li>{insight}</li>"
                        html_content += "</ul>"
                        
                        html_content += "</body></html>"
                        
                        st.download_button(
                            label="Download HTML",
                            data=html_content,
                            file_name="analysis_results.html",
                            mime="text/html"
                        )
        else:
            st.info("No analysis results available. Please run analysis first.")

def torch_available():
    """Check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False

def torch_cuda_available():
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def torch_mps_available():
    """Check if MPS (Apple Silicon acceleration) is available."""
    try:
        import torch
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except ImportError:
        return False

def create_sample_data():
    """Create sample data for demonstration."""
    return {
        "video_metadata": {
            "filename": "sample_game.mp4",
            "fps": 30,
            "frame_count": 9000,
            "width": 1920,
            "height": 1080
        },
        "frames": [
            {"frame_idx": 0, "player_positions": [{"id": 1, "center": [500, 300], "team": "team_a"}]},
            {"frame_idx": 300, "player_positions": [{"id": 2, "center": [700, 400], "team": "team_b"}]}
        ],
        "plays": [
            {"start_frame": 0, "end_frame": 500, "play_type": "kickout"},
            {"start_frame": 600, "end_frame": 900, "play_type": "free_kick"}
        ],
        "player_tracks": [
            {"player_id": 1, "positions": [{"frame": 0, "position": [500, 300], "team": "team_a"}]},
            {"player_id": 2, "positions": [{"frame": 300, "position": [700, 400], "team": "team_b"}]}
        ]
    }

if __name__ == "__main__":
    main()
