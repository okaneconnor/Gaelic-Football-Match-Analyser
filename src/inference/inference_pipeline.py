"""
Inference Pipeline Module for Gaelic Football Match Analyser

This module handles:
1. Running the fine-tuned LLM on new game footage
2. Processing video or pre-processed data inputs
3. Generating analytical outputs and insights
"""

import logging
import os
import json
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import time

from src.data_processing.video_processor import VideoProcessor
from src.data_processing.data_transformer import DataTransformer

logger = logging.getLogger(__name__)

class InferencePipeline:
    """
    Run inference on new game footage using fine-tuned LLM.
    """
    
    def __init__(self, 
                 model,
                 max_new_tokens: int = 300,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 chunk_size: int = 5,
                 output_dir: Optional[str] = None):
        """
        Initialize the inference pipeline.
        
        Args:
            model: Fine-tuned model to use for inference
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p probability threshold for sampling
            chunk_size: Number of plays to process at once
            output_dir: Directory to save outputs
        """
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.chunk_size = chunk_size
        self.output_dir = output_dir
        
        # Get tokenizer from model if available
        self.tokenizer = getattr(model, "tokenizer", None)
        if self.tokenizer is None and hasattr(model, "config") and hasattr(model.config, "_name_or_path"):
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
                logger.info(f"Loaded tokenizer from {model.config._name_or_path}")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer: {e}")
                logger.warning("You'll need to provide a tokenizer when running inference")
    
    def run(self, 
            input_data: Union[str, Dict[str, Any]], 
            tokenizer = None,
            output_type: str = "json") -> Dict[str, Any]:
        """
        Run inference on input data.
        
        Args:
            input_data: Either a path to a video file or pre-processed data
            tokenizer: Tokenizer to use (if not already set)
            output_type: Type of output to generate ('json', 'text', or 'both')
            
        Returns:
            Dict containing inference results
        """
        logger.info("Starting inference pipeline")
        
        # Set tokenizer if provided
        if tokenizer is not None:
            self.tokenizer = tokenizer
        
        # Check that tokenizer exists
        if self.tokenizer is None:
            raise ValueError("No tokenizer available. Please provide a tokenizer when running inference")
        
        # Create output directory if needed
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Process input data
        processed_data = self._process_input(input_data)
        
        # Run inference
        results = self._run_inference(processed_data, output_type)
        
        # Save results
        if self.output_dir:
            output_path = os.path.join(self.output_dir, "inference_results.json")
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved inference results to {output_path}")
        
        return results
    
    def _process_input(self, input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process input data (video or pre-processed data).
        
        Args:
            input_data: Either a path to a video file or pre-processed data
            
        Returns:
            Processed data ready for inference
        """
        # Check if input is a video path
        if isinstance(input_data, str) and os.path.exists(input_data):
            file_ext = os.path.splitext(input_data)[1].lower()
            if file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                logger.info(f"Processing video: {input_data}")
                
                # Process video
                video_processor = VideoProcessor()
                video_data = video_processor.process(input_data, output_dir=self.output_dir)
                
                # Transform video data
                data_transformer = DataTransformer()
                processed_data = data_transformer.transform(video_data, output_dir=self.output_dir)
                
                return processed_data
                
            elif file_ext in ['.json', '.jsonl']:
                # Load pre-processed data from JSON file
                logger.info(f"Loading pre-processed data from: {input_data}")
                with open(input_data, 'r') as f:
                    processed_data = json.load(f)
                return processed_data
            
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
        
        # Check if input is already processed data
        elif isinstance(input_data, dict):
            logger.info("Using provided pre-processed data")
            return input_data
        
        else:
            raise ValueError("Input must be either a video file path or pre-processed data")
    
    def _run_inference(self, processed_data: Dict[str, Any], output_type: str) -> Dict[str, Any]:
        """
        Run LLM inference on processed data.
        
        Args:
            processed_data: Processed data to run inference on
            output_type: Type of output to generate
            
        Returns:
            Dict containing inference results
        """
        # Check that required data is present
        if "plays" not in processed_data:
            raise ValueError("Processed data must contain 'plays'")
        
        plays = processed_data["plays"]
        logger.info(f"Running inference on {len(plays)} plays")
        
        # Prepare results structure
        results = {
            "metadata": {
                "source": processed_data.get("metadata", {}).get("source_video", "unknown"),
                "inference_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": str(type(self.model).__name__),
                "parameters": {
                    "max_new_tokens": self.max_new_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p
                }
            },
            "play_analyses": [],
            "game_analysis": {},
            "strategic_insights": []
        }
        
        # Process plays in chunks to avoid memory issues
        for i in range(0, len(plays), self.chunk_size):
            chunk = plays[i:i+self.chunk_size]
            logger.info(f"Processing plays {i+1}-{i+len(chunk)} of {len(plays)}")
            
            # Analyze each play in the chunk
            for play in chunk:
                play_analysis = self._analyze_play(play)
                results["play_analyses"].append(play_analysis)
        
        # Generate overall game analysis
        results["game_analysis"] = self._analyze_game(processed_data, results["play_analyses"])
        
        # Generate strategic insights
        results["strategic_insights"] = self._generate_strategic_insights(
            processed_data, results["play_analyses"], results["game_analysis"]
        )
        
        return results
    
    def _analyze_play(self, play: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a single play using the LLM.
        
        Args:
            play: Play data to analyze
            
        Returns:
            Dict containing play analysis
        """
        # Get play description
        description = play.get("description", "No description available")
        play_type = play.get("play_type", "unknown")
        
        # Create prompts for different analysis types
        prompts = {
            "basic_analysis": f"### Instruction: Analyze this {play_type} play in a Gaelic football match: {description}\n\n### Response:",
            "tactical_breakdown": f"### Instruction: Provide a tactical breakdown of this {play_type} play: {description}\n\n### Response:",
            "player_movement": f"### Instruction: Analyze the player movements in this {play_type} play: {description}\n\n### Response:"
        }
        
        analysis_results = {}
        
        # Generate responses for each prompt type
        for analysis_type, prompt in prompts.items():
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # Move to same device as model
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True
                )
            
            # Decode
            output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Extract response part
            response = output_text.split("### Response:")[-1].strip()
            
            # Add to results
            analysis_results[analysis_type] = response
        
        # Compile final play analysis
        play_analysis = {
            "play_id": play.get("start_frame", 0),
            "play_type": play_type,
            "description": description,
            "analysis": analysis_results["basic_analysis"],
            "tactical_breakdown": analysis_results["tactical_breakdown"],
            "player_movement_analysis": analysis_results["player_movement"],
            "key_insights": self._extract_key_insights(analysis_results)
        }
        
        return play_analysis
    
    def _extract_key_insights(self, analysis_results: Dict[str, str]) -> List[str]:
        """
        Extract key insights from analysis texts.
        
        Args:
            analysis_results: Dict of analysis results by type
            
        Returns:
            List of key insights
        """
        # In a real implementation, this would use more sophisticated
        # extraction logic. For now, we'll generate a simple prompt
        # to extract key points.
        
        combined_analysis = "\n\n".join(analysis_results.values())
        
        prompt = f"### Instruction: Extract the 3-5 most important insights from this football play analysis as a bullet point list:\n\n{combined_analysis}\n\n### Response:"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Move to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.5,  # Lower temperature for more focused output
                top_p=0.9,
                do_sample=True
            )
        
        # Decode
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract response part
        response = output_text.split("### Response:")[-1].strip()
        
        # Split into bullet points
        insights = [
            point.strip().lstrip("-•*").strip()
            for point in response.split("\n")
            if point.strip() and any(point.strip().startswith(c) for c in "-•*")
        ]
        
        # If no bullet points found, try splitting by sentences
        if not insights:
            import re
            insights = [s.strip() for s in re.split(r'(?<=[.!?])\s+', response) if s.strip()]
            insights = insights[:5]  # Limit to 5 insights
        
        return insights
    
    def _analyze_game(self, 
                     processed_data: Dict[str, Any], 
                     play_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate overall game analysis.
        
        Args:
            processed_data: Processed game data
            play_analyses: List of play analyses
            
        Returns:
            Dict containing game analysis
        """
        # Combine all play descriptions and analyses
        all_descriptions = []
        all_analyses = []
        
        for play in processed_data.get("plays", []):
            if "description" in play:
                all_descriptions.append(play["description"])
        
        for analysis in play_analyses:
            all_analyses.append(analysis["analysis"])
        
        combined_text = "\n\n".join(all_descriptions + all_analyses)
        
        # Limit length to avoid context overflow
        if len(combined_text) > 8000:
            combined_text = combined_text[:8000] + "..."
        
        # Create prompt for game analysis
        prompt = f"### Instruction: Provide a comprehensive analysis of this Gaelic football game based on the following play descriptions and analyses:\n\n{combined_text}\n\n### Response:"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Move to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens * 2,  # Longer output for game analysis
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True
            )
        
        # Decode
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract response part
        response = output_text.split("### Response:")[-1].strip()
        
        # Create game analysis structure
        game_analysis = {
            "overview": response,
            "team_performance": self._analyze_team_performance(processed_data, play_analyses),
            "key_moments": self._identify_key_moments(play_analyses),
            "statistics": self._generate_statistics(processed_data, play_analyses)
        }
        
        return game_analysis
    
    def _analyze_team_performance(self, 
                                 processed_data: Dict[str, Any], 
                                 play_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze team performance from play analyses.
        
        Args:
            processed_data: Processed game data
            play_analyses: List of play analyses
            
        Returns:
            Dict containing team performance analysis
        """
        # Extract team-specific information
        teams = ["Team A", "Team B"]  # In a real implementation, get actual team names
        
        # Combine all analyses
        all_analyses = "\n\n".join([a["analysis"] for a in play_analyses])
        
        team_analyses = {}
        
        for team in teams:
            # Create prompt for team analysis
            prompt = f"### Instruction: Analyze the performance of {team} in this Gaelic football game based on the following play analyses:\n\n{all_analyses}\n\n### Response:"
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # Move to same device as model
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True
                )
            
            # Decode
            output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Extract response part
            response = output_text.split("### Response:")[-1].strip()
            
            team_analyses[team] = response
        
        return team_analyses
    
    def _identify_key_moments(self, play_analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identify key moments from play analyses.
        
        Args:
            play_analyses: List of play analyses
            
        Returns:
            List of key moments
        """
        # For simplicity, we'll select plays with the most insights
        play_analyses_sorted = sorted(
            play_analyses, 
            key=lambda x: len(x.get("key_insights", [])), 
            reverse=True
        )
        
        key_moments = []
        
        # Take top 5 plays or fewer if there aren't that many
        for i, play in enumerate(play_analyses_sorted[:5]):
            key_moments.append({
                "play_id": play.get("play_id", i),
                "play_type": play.get("play_type", "unknown"),
                "description": play.get("description", ""),
                "significance": self._generate_significance(play),
                "key_insights": play.get("key_insights", [])
            })
        
        return key_moments
    
    def _generate_significance(self, play: Dict[str, Any]) -> str:
        """
        Generate a description of play significance.
        
        Args:
            play: Play analysis
            
        Returns:
            String describing play significance
        """
        # Combine analysis texts
        analysis_text = play.get("analysis", "")
        
        # Create prompt for significance
        prompt = f"### Instruction: In one sentence, explain why this play is significant to the outcome of the Gaelic football match:\n\n{analysis_text}\n\n### Response:"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Move to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=50,  # Short response
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        # Decode
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract response part
        response = output_text.split("### Response:")[-1].strip()
        
        return response
    
    def _generate_statistics(self, 
                            processed_data: Dict[str, Any], 
                            play_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate statistical summary from play analyses.
        
        Args:
            processed_data: Processed game data
            play_analyses: List of play analyses
            
        Returns:
            Dict containing statistical summary
        """
        # Count play types
        play_types = {}
        for play in processed_data.get("plays", []):
            play_type = play.get("play_type", "unknown")
            play_types[play_type] = play_types.get(play_type, 0) + 1
        
        # In a real implementation, extract more statistics from the data
        # For now, we'll use placeholder statistics
        statistics = {
            "play_types": play_types,
            "team_scores": {
                "Team A": {"points": 12, "goals": 2},
                "Team B": {"points": 9, "goals": 1}
            },
            "possession": {
                "Team A": 55,
                "Team B": 45
            },
            "scoring_efficiency": {
                "Team A": 65,
                "Team B": 58
            }
        }
        
        return statistics
    
    def _generate_strategic_insights(self, 
                                    processed_data: Dict[str, Any], 
                                    play_analyses: List[Dict[str, Any]], 
                                    game_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate strategic insights from analyses.
        
        Args:
            processed_data: Processed game data
            play_analyses: List of play analyses
            game_analysis: Game analysis
            
        Returns:
            List of strategic insights
        """
        # Combine game analysis and key insights from plays
        game_overview = game_analysis.get("overview", "")
        team_performances = "\n\n".join(game_analysis.get("team_performance", {}).values())
        
        key_insights_text = ""
        for play in play_analyses:
            insights = play.get("key_insights", [])
            if insights:
                key_insights_text += f"Play {play.get('play_id', '')}: {'. '.join(insights)}\n\n"
        
        combined_text = f"{game_overview}\n\n{team_performances}\n\n{key_insights_text}"
        
        # Limit length to avoid context overflow
        if len(combined_text) > 8000:
            combined_text = combined_text[:8000] + "..."
        
        # Create prompt for strategic insights
        prompt = f"### Instruction: Based on the following game analysis, generate 5 strategic insights that would be valuable for coaching staff:\n\n{combined_text}\n\n### Response:"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Move to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True
            )
        
        # Decode
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract response part
        response = output_text.split("### Response:")[-1].strip()
        
        # Parse insights
        insights = []
        
        # Try to find bullet points or numbered items
        import re
        bullet_pattern = r'(?:^|\n)(?:\d+\.|[-•*])\s+(.*?)(?=(?:\n(?:\d+\.|[-•*])\s+|\n\n|$))'
        matches = re.findall(bullet_pattern, response, re.DOTALL)
        
        if matches:
            # Found structured insights
            for i, match in enumerate(matches):
                insights.append({
                    "id": i + 1,
                    "insight": match.strip(),
                    "category": self._categorize_insight(match.strip()),
                    "application": self._generate_insight_application(match.strip())
                })
        else:
            # Split by paragraphs
            paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
            for i, paragraph in enumerate(paragraphs[:5]):  # Limit to 5 insights
                insights.append({
                    "id": i + 1,
                    "insight": paragraph,
                    "category": self._categorize_insight(paragraph),
                    "application": self._generate_insight_application(paragraph)
                })
        
        return insights
    
    def _categorize_insight(self, insight_text: str) -> str:
        """
        Categorize an insight based on its content.
        
        Args:
            insight_text: Text of the insight
            
        Returns:
            Category name
        """
        # Create prompt for categorization
        prompt = f"### Instruction: Categorize this Gaelic football insight into exactly one of these categories: 'Offensive Strategy', 'Defensive Strategy', 'Player Development', 'Team Formation', or 'Game Management':\n\n{insight_text}\n\n### Response:"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Move to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=20,  # Very short response
                temperature=0.3,  # Low temperature for more precise categorization
                top_p=0.9,
                do_sample=True
            )
        
        # Decode
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract response part
        response = output_text.split("### Response:")[-1].strip()
        
        # Map to predefined categories
        categories = ["Offensive Strategy", "Defensive Strategy", "Player Development", "Team Formation", "Game Management"]
        
        for category in categories:
            if category.lower() in response.lower():
                return category
        
        # Default
        return "General Strategy"
    
    def _generate_insight_application(self, insight_text: str) -> str:
        """
        Generate practical application advice for an insight.
        
        Args:
            insight_text: Text of the insight
            
        Returns:
            Practical application advice
        """
        # Create prompt for application
        prompt = f"### Instruction: Provide a brief, practical way to apply this Gaelic football insight in training or gameplay:\n\n{insight_text}\n\n### Response:"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Move to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=100,  # Concise response
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        # Decode
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract response part
        response = output_text.split("### Response:")[-1].strip()
        
        return response
