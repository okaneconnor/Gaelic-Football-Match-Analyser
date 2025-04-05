"""
Data Transformation Module for Gaelic Football Match Analyser

This module handles:
1. Conversion of video data into structured text descriptions
2. Normalization of play descriptions
3. Tagging important events
4. Creating training examples for LLM fine-tuning
"""

import logging
import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class DataTransformer:
    """
    Transform processed video data into a format suitable for LLM consumption.
    """
    
    def __init__(self, 
                 vocabulary_path: Optional[str] = None,
                 template_path: Optional[str] = None):
        """
        Initialize the data transformer.
        
        Args:
            vocabulary_path: Path to football/Gaelic football vocabulary file
            template_path: Path to description templates file
        """
        self.vocabulary = self._load_vocabulary(vocabulary_path)
        self.templates = self._load_templates(template_path)
        
    def _load_vocabulary(self, path: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Load football/Gaelic football vocabulary from file or use default.
        
        Args:
            path: Path to vocabulary file
            
        Returns:
            Dictionary of terms categorized by type
        """
        if path and os.path.exists(path):
            logger.info(f"Loading vocabulary from {path}")
            with open(path, 'r') as f:
                return json.load(f)
        
        # Default vocabulary
        logger.info("Using default football/Gaelic football vocabulary")
        return {
            "play_types": [
                "kickout", "free_kick", "sideline_kick", "penalty", "open_play",
                "handpass", "solo_run", "tackle", "mark", "high_catch"
            ],
            "positions": [
                "goalkeeper", "full_back", "right_corner_back", "left_corner_back",
                "center_half_back", "right_half_back", "left_half_back",
                "midfield", "center_half_forward", "right_half_forward", 
                "left_half_forward", "full_forward", "right_corner_forward",
                "left_corner_forward"
            ],
            "actions": [
                "catches", "kicks", "handpasses", "solos", "tackles", "marks",
                "blocks", "scores", "shoots", "defends", "attacks", "intercepts",
                "passes", "runs"
            ],
            "outcomes": [
                "point", "goal", "wide", "short", "turnover", "foul", "dispossession"
            ],
            "formations": [
                "defensive", "attacking", "press", "counter_attack", "zonal", "man_to_man"
            ]
        }
    
    def _load_templates(self, path: Optional[str] = None) -> List[str]:
        """
        Load description templates from file or use default.
        
        Args:
            path: Path to templates file
            
        Returns:
            List of template strings
        """
        if path and os.path.exists(path):
            logger.info(f"Loading templates from {path}")
            with open(path, 'r') as f:
                return json.load(f)
        
        # Default templates
        logger.info("Using default description templates")
        return [
            "A {team} player performs a {action} from the {position} position.",
            "The {position} {action} the ball, resulting in a {outcome}.",
            "From a {play_type}, the {team} {position} {action} and the play develops with a {formation} formation.",
            "After receiving the ball, the {position} {action} through the {opposition_team}'s defense.",
            "The {team} sets up in a {formation} formation as the {position} prepares to {action}.",
            "A {outcome} is scored by the {position} after a well-executed {play_type}.",
            "The {opposition_team} defense uses a {formation} to prevent the {team} {position} from {action}.",
            "In a key moment, the {position} {action}, leading to a {outcome} for {team}."
        ]
        
    def transform(self, video_data: Dict[str, Any], output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Transform processed video data into a format suitable for LLM.
        
        Args:
            video_data: Processed video data from VideoProcessor
            output_dir: Directory to save transformed data
            
        Returns:
            Dict containing transformed data
        """
        logger.info("Transforming video data for LLM processing")
        
        # Get metadata
        if "video_metadata" not in video_data:
            raise ValueError("Invalid video data: missing video_metadata")
        
        metadata = video_data["video_metadata"]
        
        # Create output structure
        output_data = {
            "metadata": {
                "source_video": metadata.get("filename", "unknown"),
                "transformed_format": "text_descriptions",
                "vocabulary_size": sum(len(v) for v in self.vocabulary.values()),
                "template_count": len(self.templates)
            },
            "plays": [],
            "global_patterns": [],
            "training_examples": []
        }
        
        # Process each play
        for play in video_data.get("plays", []):
            play_description = self._generate_play_description(play, video_data)
            
            transformed_play = {
                "start_frame": play.get("start_frame"),
                "end_frame": play.get("end_frame"),
                "play_type": play.get("play_type"),
                "description": play_description,
                "tokens": self._tokenize_description(play_description),
                "key_entities": self._extract_key_entities(play_description)
            }
            
            output_data["plays"].append(transformed_play)
            
            # Create training examples for this play
            examples = self._create_training_examples(transformed_play)
            output_data["training_examples"].extend(examples)
        
        # Find global patterns
        output_data["global_patterns"] = self._find_global_patterns(output_data["plays"])
        
        # Save transformed data if output_dir specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "transformed_data.json")
            
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
                
            logger.info(f"Saved transformed data to {output_path}")
            
            # Also save training examples as jsonl for easy model fine-tuning
            examples_path = os.path.join(output_dir, "training_examples.jsonl")
            with open(examples_path, 'w') as f:
                for example in output_data["training_examples"]:
                    f.write(json.dumps(example) + '\n')
                    
            logger.info(f"Saved {len(output_data['training_examples'])} training examples to {examples_path}")
        
        return output_data
    
    def _generate_play_description(self, play: Dict[str, Any], video_data: Dict[str, Any]) -> str:
        """
        Generate a textual description of a play based on video data.
        
        Args:
            play: Play data
            video_data: Complete video data
            
        Returns:
            Textual description of the play
        """
        # In a real implementation, this would use the player tracking data
        # to generate a detailed description of the play. For this example,
        # we'll use templates with some randomization.
        
        play_type = play.get("play_type", "unknown")
        
        # Select random values for template placeholders
        team = np.random.choice(["Team A", "Team B"])
        opposition_team = "Team B" if team == "Team A" else "Team A"
        position = np.random.choice(self.vocabulary["positions"])
        action = np.random.choice(self.vocabulary["actions"])
        outcome = np.random.choice(self.vocabulary["outcomes"])
        formation = np.random.choice(self.vocabulary["formations"])
        
        # Select a random template
        template = np.random.choice(self.templates)
        
        # Fill in the template
        description = template.format(
            play_type=play_type,
            team=team,
            opposition_team=opposition_team,
            position=position,
            action=action,
            outcome=outcome,
            formation=formation
        )
        
        # Add some context about the play
        context = f"This {play_type} occurred at frame {play.get('start_frame')}. "
        
        # Add context about the score if applicable
        if outcome in ["point", "goal"]:
            points = 1 if outcome == "point" else 3
            context += f"This resulted in {points} point(s) for {team}. "
        
        return context + description
    
    def _tokenize_description(self, description: str) -> List[str]:
        """
        Simple tokenization of a description into words.
        
        Args:
            description: Play description
            
        Returns:
            List of tokens
        """
        # This is a very basic tokenizer - in a real implementation,
        # you would use a proper NLP tokenizer
        return description.lower().replace(".", "").replace(",", "").split()
    
    def _extract_key_entities(self, description: str) -> Dict[str, List[str]]:
        """
        Extract key entities from a play description.
        
        Args:
            description: Play description
            
        Returns:
            Dict of entity types and extracted entities
        """
        # This is a simplified entity extraction - in a real implementation,
        # you would use NER or other NLP techniques
        
        entities = {
            "teams": [],
            "players": [],
            "actions": [],
            "outcomes": []
        }
        
        # Simple string matching against vocabulary
        description_lower = description.lower()
        
        # Check for team names
        for team in ["team a", "team b"]:
            if team in description_lower:
                entities["teams"].append(team)
        
        # Check for positions (as proxy for players)
        for position in self.vocabulary["positions"]:
            if position in description_lower:
                entities["players"].append(position)
        
        # Check for actions
        for action in self.vocabulary["actions"]:
            if action in description_lower:
                entities["actions"].append(action)
        
        # Check for outcomes
        for outcome in self.vocabulary["outcomes"]:
            if outcome in description_lower:
                entities["outcomes"].append(outcome)
        
        return entities
    
    def _create_training_examples(self, transformed_play: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create training examples for LLM fine-tuning based on a transformed play.
        
        Args:
            transformed_play: Transformed play data
            
        Returns:
            List of training examples
        """
        examples = []
        description = transformed_play["description"]
        
        # Create a Q&A example
        examples.append({
            "input": f"Describe the following play in a Gaelic football match: {transformed_play['play_type']}",
            "output": description,
            "type": "description_generation"
        })
        
        # Create an analysis example
        if "key_entities" in transformed_play:
            entities = transformed_play["key_entities"]
            if entities.get("outcomes"):
                outcome = entities["outcomes"][0] if entities["outcomes"] else "unknown"
                examples.append({
                    "input": f"Analyze why this play resulted in a {outcome}: {description}",
                    "output": f"The play resulted in a {outcome} because of effective positioning and execution by the {entities.get('teams', ['team'])[0]} player in the {entities.get('players', ['unknown'])[0]} position. The key action was {entities.get('actions', ['unknown'])[0]}.",
                    "type": "play_analysis"
                })
        
        # Create a pattern recognition example
        examples.append({
            "input": f"Identify patterns in this play: {description}",
            "output": f"This play demonstrates a common pattern where a {transformed_play['play_type']} leads to a scoring opportunity. The movement of players from defensive to attacking positions is characteristic of successful plays.",
            "type": "pattern_recognition"
        })
        
        return examples
    
    def _find_global_patterns(self, plays: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identify global patterns across multiple plays.
        
        Args:
            plays: List of transformed plays
            
        Returns:
            List of identified patterns
        """
        # This would use more sophisticated analysis in a real implementation
        # For now, we'll generate some sample patterns
        
        # Count play types
        play_types = {}
        for play in plays:
            play_type = play.get("play_type", "unknown")
            if play_type in play_types:
                play_types[play_type] += 1
            else:
                play_types[play_type] = 1
        
        patterns = []
        
        # Find most common play type
        if play_types:
            most_common = max(play_types.items(), key=lambda x: x[1])
            patterns.append({
                "pattern_type": "frequency",
                "description": f"The most common play type is {most_common[0]}, occurring {most_common[1]} times.",
                "frequency": most_common[1],
                "play_type": most_common[0]
            })
        
        # Add some fictional patterns as examples
        patterns.extend([
            {
                "pattern_type": "sequence",
                "description": "Kickouts followed by handpasses often lead to quick counter-attacks.",
                "confidence": 0.8
            },
            {
                "pattern_type": "formation",
                "description": "Defensive formations are more common in the final quarter of the game.",
                "confidence": 0.7
            },
            {
                "pattern_type": "scoring",
                "description": "Points are more frequently scored from the right side of the field.",
                "confidence": 0.65
            }
        ])
        
        return patterns
