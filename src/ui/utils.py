"""
UI Utilities Module for Gaelic Football Match Analyser

This module contains utility functions for the Streamlit UI.
"""

import base64
import time
from typing import Dict, List, Any, Optional, Union

def format_seconds(seconds: float) -> str:
    """
    Format seconds into a human-readable string.
    
    Args:
        seconds: Number of seconds
        
    Returns:
        Formatted string (e.g. "2m 30s" or "1h 15m 30s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {int(seconds)}s"
    
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

def create_download_link(content: Union[str, bytes], filename: str, mime_type: str) -> str:
    """
    Create a download link for Streamlit.
    
    Args:
        content: File content (string or bytes)
        filename: Name of the file to download
        mime_type: MIME type of the file
        
    Returns:
        HTML string with download link
    """
    if isinstance(content, str):
        content = content.encode()
    
    b64 = base64.b64encode(content).decode()
    href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

def humanize_bytes(num_bytes: int) -> str:
    """
    Convert bytes to a human-readable string.
    
    Args:
        num_bytes: Number of bytes
        
    Returns:
        Human-readable string (e.g. "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"

def format_percent(value: float, decimals: int = 1) -> str:
    """
    Format a value as a percentage string.
    
    Args:
        value: Value to format (0.0 to 1.0)
        decimals: Number of decimal places
        
    Returns:
        Percentage string (e.g. "75.5%")
    """
    return f"{value * 100:.{decimals}f}%"

def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text with ellipsis if it exceeds max_length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."

def get_color_for_value(value: float, 
                       min_value: float = 0.0, 
                       max_value: float = 1.0, 
                       colors: List[str] = None) -> str:
    """
    Get a color for a value based on its position in a range.
    
    Args:
        value: Value to get color for
        min_value: Minimum value in range
        max_value: Maximum value in range
        colors: List of colors to interpolate between
        
    Returns:
        CSS color string
    """
    if colors is None:
        colors = ["#ff0000", "#ffff00", "#00ff00"]  # Red, Yellow, Green
    
    # Normalize value to 0-1 range
    normalized = (value - min_value) / (max_value - min_value)
    normalized = max(0, min(1, normalized))  # Clamp to 0-1
    
    # Determine which color pair to interpolate between
    num_colors = len(colors)
    segment_size = 1.0 / (num_colors - 1)
    segment = int(normalized / segment_size)
    segment = min(segment, num_colors - 2)  # Ensure we don't go out of bounds
    
    # Calculate position within the segment
    segment_pos = (normalized - segment * segment_size) / segment_size
    
    # Parse colors
    import re
    color1 = colors[segment]
    color2 = colors[segment + 1]
    
    # Parse hex colors to rgb
    r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
    r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)
    
    # Interpolate
    r = int(r1 + segment_pos * (r2 - r1))
    g = int(g1 + segment_pos * (g2 - g1))
    b = int(b1 + segment_pos * (b2 - b1))
    
    return f"#{r:02x}{g:02x}{b:02x}"

def get_model_emoji(model_name: str) -> str:
    """
    Get an emoji representing a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Emoji string
    """
    model_name = model_name.lower()
    if "llama" in model_name:
        return "🦙"
    elif "mistral" in model_name:
        return "🌪️"
    elif "phi" in model_name:
        return "φ"
    elif "gpt" in model_name:
        return "🤖"
    else:
        return "🧠"

def format_timestamp(timestamp: Union[float, int]) -> str:
    """
    Format a Unix timestamp into a human-readable string.
    
    Args:
        timestamp: Unix timestamp
        
    Returns:
        Formatted string
    """
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))

def create_comparison_table(models_data: Dict[str, Dict[str, Any]]) -> str:
    """
    Create an HTML table for model comparison.
    
    Args:
        models_data: Dict of model data
        
    Returns:
        HTML string
    """
    html = """
    <style>
        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            font-family: Arial, sans-serif;
        }
        .comparison-table th, .comparison-table td {
            padding: 8px;
            text-align: left;
            border: 1px solid #ddd;
        }
        .comparison-table th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        .comparison-table tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .comparison-table tr:hover {
            background-color: #e9e9e9;
        }
        .success {
            color: green;
        }
        .failure {
            color: red;
        }
    </style>
    <table class="comparison-table">
        <tr>
            <th>Model</th>
            <th>Load Time</th>
            <th>Inference Time</th>
            <th>Device</th>
            <th>Status</th>
        </tr>
    """
    
    for model_name, data in models_data.items():
        success = data.get("success", False)
        status_class = "success" if success else "failure"
        status_text = "Success" if success else f"Failed: {data.get('error', 'Unknown error')}"
        
        html += f"""
        <tr>
            <td>{get_model_emoji(model_name)} {model_name}</td>
            <td>{format_seconds(data.get('load_time_seconds', 0)) if success else '-'}</td>
            <td>{format_seconds(data.get('inference_time_seconds', 0)) if success else '-'}</td>
            <td>{data.get('device', '-')}</td>
            <td class="{status_class}">{status_text}</td>
        </tr>
        """
    
    html += "</table>"
    return html
