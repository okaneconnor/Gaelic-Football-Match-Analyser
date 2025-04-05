"""
Analysis Module for Gaelic Football Match Analyser

This module handles:
1. Post-processing of inference results
2. Pattern extraction and trend identification
3. Visualization and report generation
"""

import logging
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import time

logger = logging.getLogger(__name__)

class Analyzer:
    """
    Analyze inference results to extract patterns, trends, and insights.
    """
    
    def __init__(self, 
                 output_dir: Optional[str] = None,
                 visualize: bool = True,
                 export_formats: List[str] = ["json", "html"]):
        """
        Initialize the analyzer.
        
        Args:
            output_dir: Directory to save analysis outputs
            visualize: Whether to generate visualizations
            export_formats: Formats to export analysis results in
        """
        self.output_dir = output_dir
        self.visualize = visualize
        self.export_formats = export_formats
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    def analyze(self, inference_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze inference results to extract patterns and insights.
        
        Args:
            inference_results: Results from inference pipeline
            
        Returns:
            Dict containing analysis results
        """
        logger.info("Analyzing inference results")
        
        # Validate input
        if not inference_results.get("play_analyses"):
            raise ValueError("Inference results must contain play_analyses")
        
        # Initialize result structure
        analysis_results = {
            "metadata": {
                "source": inference_results.get("metadata", {}).get("source", "unknown"),
                "analysis_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "play_count": len(inference_results.get("play_analyses", [])),
            },
            "patterns": [],
            "trends": {},
            "recommendations": [],
            "visualizations": {}
        }
        
        # Extract patterns
        analysis_results["patterns"] = self._extract_patterns(inference_results)
        
        # Identify trends
        analysis_results["trends"] = self._identify_trends(inference_results)
        
        # Generate recommendations
        analysis_results["recommendations"] = self._generate_recommendations(
            inference_results, analysis_results["patterns"], analysis_results["trends"]
        )
        
        # Generate visualizations if requested
        if self.visualize:
            analysis_results["visualizations"] = self._generate_visualizations(
                inference_results, analysis_results
            )
        
        # Export results
        if self.output_dir:
            self._export_results(analysis_results)
        
        logger.info(f"Analysis complete. Identified {len(analysis_results['patterns'])} patterns")
        
        return analysis_results
    
    def _extract_patterns(self, inference_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract patterns from inference results.
        
        Args:
            inference_results: Results from inference pipeline
            
        Returns:
            List of identified patterns
        """
        play_analyses = inference_results.get("play_analyses", [])
        strategic_insights = inference_results.get("strategic_insights", [])
        
        patterns = []
        
        # Extract common themes from key insights
        all_insights = []
        for play in play_analyses:
            all_insights.extend(play.get("key_insights", []))
        
        # Count keyword occurrences
        keywords = {}
        for insight in all_insights:
            words = insight.lower().split()
            for word in words:
                if len(word) > 4:  # Filter out short words
                    keywords[word] = keywords.get(word, 0) + 1
        
        # Find top keywords
        top_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Add keyword patterns
        for keyword, count in top_keywords:
            patterns.append({
                "type": "keyword",
                "keyword": keyword,
                "frequency": count,
                "description": f"The term '{keyword}' appears frequently in play insights ({count} times)",
                "confidence": min(count / len(all_insights) * 5, 1.0)  # Scale confidence based on frequency
            })
        
        # Extract play type patterns
        play_types = {}
        for play in play_analyses:
            play_type = play.get("play_type", "unknown")
            play_types[play_type] = play_types.get(play_type, 0) + 1
        
        # Add play type patterns
        for play_type, count in play_types.items():
            if count > 1:  # Only add if it appears multiple times
                patterns.append({
                    "type": "play_type",
                    "play_type": play_type,
                    "frequency": count,
                    "description": f"The play type '{play_type}' appears {count} times",
                    "confidence": min(count / len(play_analyses) * 2, 1.0)
                })
        
        # Extract patterns from strategic insights
        for insight in strategic_insights:
            patterns.append({
                "type": "strategic",
                "category": insight.get("category", "General"),
                "description": insight.get("insight", ""),
                "application": insight.get("application", ""),
                "confidence": 0.8  # Strategic insights are generally high confidence
            })
        
        # Sort patterns by confidence
        patterns = sorted(patterns, key=lambda x: x.get("confidence", 0), reverse=True)
        
        return patterns
    
    def _identify_trends(self, inference_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify trends in play analyses.
        
        Args:
            inference_results: Results from inference pipeline
            
        Returns:
            Dict of identified trends
        """
        play_analyses = inference_results.get("play_analyses", [])
        game_analysis = inference_results.get("game_analysis", {})
        
        trends = {
            "play_type_distribution": {},
            "team_performance": {},
            "temporal_patterns": []
        }
        
        # Play type distribution
        play_types = {}
        for play in play_analyses:
            play_type = play.get("play_type", "unknown")
            play_types[play_type] = play_types.get(play_type, 0) + 1
        
        trends["play_type_distribution"] = {
            "counts": play_types,
            "percentages": {pt: count / len(play_analyses) * 100 for pt, count in play_types.items()}
        }
        
        # Team performance (extract from game analysis)
        if "team_performance" in game_analysis:
            team_perf = game_analysis["team_performance"]
            trends["team_performance"] = {
                "analysis": team_perf,
                "summary": "Analysis of team performances is available in team_performance field"
            }
        
        # Temporal patterns (plays over time)
        if len(play_analyses) > 3:
            # Look for patterns in sequence
            for i in range(len(play_analyses) - 2):
                three_plays = play_analyses[i:i+3]
                play_types = [p.get("play_type", "unknown") for p in three_plays]
                
                # Check if all three plays are of the same type
                if len(set(play_types)) == 1:
                    trends["temporal_patterns"].append({
                        "pattern_type": "sequence",
                        "description": f"Three consecutive {play_types[0]} plays detected",
                        "start_index": i,
                        "play_ids": [p.get("play_id", j+i) for j, p in enumerate(three_plays)]
                    })
        
        # Add statistics from game analysis if available
        if "statistics" in game_analysis:
            trends["statistics"] = game_analysis["statistics"]
        
        return trends
    
    def _generate_recommendations(self, 
                                 inference_results: Dict[str, Any],
                                 patterns: List[Dict[str, Any]],
                                 trends: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate actionable recommendations based on analysis.
        
        Args:
            inference_results: Results from inference pipeline
            patterns: Extracted patterns
            trends: Identified trends
            
        Returns:
            List of recommendations
        """
        # Start with strategic insights from inference results
        strategic_insights = inference_results.get("strategic_insights", [])
        recommendations = []
        
        # Convert strategic insights to recommendations
        for i, insight in enumerate(strategic_insights):
            if "application" in insight:
                recommendations.append({
                    "id": i + 1,
                    "category": insight.get("category", "General"),
                    "recommendation": insight.get("application", ""),
                    "based_on": insight.get("insight", ""),
                    "priority": self._calculate_priority(insight)
                })
        
        # Add recommendations based on patterns
        for pattern in patterns:
            if pattern.get("type") == "strategic":
                continue  # Already processed these
                
            if pattern.get("confidence", 0) > 0.7:
                recommendations.append({
                    "id": len(recommendations) + 1,
                    "category": "Pattern-Based",
                    "recommendation": f"Focus on {pattern.get('keyword', pattern.get('play_type', 'this aspect'))} in training and game preparation",
                    "based_on": pattern.get("description", ""),
                    "priority": "medium"
                })
        
        # Add recommendations based on trends
        if "play_type_distribution" in trends:
            # Find most common play type
            play_types = trends["play_type_distribution"].get("counts", {})
            if play_types:
                most_common = max(play_types.items(), key=lambda x: x[1])
                recommendations.append({
                    "id": len(recommendations) + 1,
                    "category": "Trend-Based",
                    "recommendation": f"Develop more counter-strategies against {most_common[0]} plays",
                    "based_on": f"{most_common[0]} is the most common play type ({most_common[1]} occurrences)",
                    "priority": "high"
                })
        
        # Sort recommendations by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations = sorted(
            recommendations, 
            key=lambda x: priority_order.get(x.get("priority", "low"), 3)
        )
        
        return recommendations
    
    def _calculate_priority(self, insight: Dict[str, Any]) -> str:
        """
        Calculate recommendation priority based on insight.
        
        Args:
            insight: Strategic insight
            
        Returns:
            Priority level (high, medium, low)
        """
        # In a real implementation, this would use more sophisticated logic
        # For now, use category to determine priority
        category = insight.get("category", "").lower()
        
        if "offensive" in category or "defensive" in category:
            return "high"
        elif "formation" in category or "player" in category:
            return "medium"
        else:
            return "low"
    
    def _generate_visualizations(self, 
                                inference_results: Dict[str, Any],
                                analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate visualizations based on analysis.
        
        Args:
            inference_results: Results from inference pipeline
            analysis_results: Analysis results
            
        Returns:
            Dict of visualization file paths
        """
        if not self.output_dir:
            logger.warning("No output directory specified, skipping visualizations")
            return {}
        
        visualizations = {}
        
        try:
            # Create visualization of play type distribution
            if "play_type_distribution" in analysis_results.get("trends", {}):
                play_types = analysis_results["trends"]["play_type_distribution"].get("counts", {})
                
                if play_types:
                    plt.figure(figsize=(10, 6))
                    plt.bar(play_types.keys(), play_types.values())
                    plt.title("Play Type Distribution")
                    plt.xlabel("Play Type")
                    plt.ylabel("Frequency")
                    plt.xticks(rotation=45)
                    
                    # Save figure
                    viz_path = os.path.join(self.output_dir, "play_type_distribution.png")
                    plt.tight_layout()
                    plt.savefig(viz_path)
                    plt.close()
                    
                    visualizations["play_type_distribution"] = viz_path
            
            # Create visualization of recommendation priorities
            recommendations = analysis_results.get("recommendations", [])
            if recommendations:
                priorities = {"high": 0, "medium": 0, "low": 0}
                for rec in recommendations:
                    priorities[rec.get("priority", "low")] += 1
                
                plt.figure(figsize=(8, 8))
                plt.pie(
                    priorities.values(), 
                    labels=priorities.keys(),
                    autopct='%1.1f%%',
                    colors=['#ff9999','#66b3ff','#99ff99']
                )
                plt.title("Recommendation Priorities")
                
                # Save figure
                viz_path = os.path.join(self.output_dir, "recommendation_priorities.png")
                plt.savefig(viz_path)
                plt.close()
                
                visualizations["recommendation_priorities"] = viz_path
            
            # Create visualization of strategic insight categories
            insights = inference_results.get("strategic_insights", [])
            if insights:
                categories = {}
                for insight in insights:
                    category = insight.get("category", "General")
                    categories[category] = categories.get(category, 0) + 1
                
                plt.figure(figsize=(10, 6))
                plt.bar(categories.keys(), categories.values())
                plt.title("Strategic Insight Categories")
                plt.xlabel("Category")
                plt.ylabel("Count")
                plt.xticks(rotation=45)
                
                # Save figure
                viz_path = os.path.join(self.output_dir, "insight_categories.png")
                plt.tight_layout()
                plt.savefig(viz_path)
                plt.close()
                
                visualizations["insight_categories"] = viz_path
                
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
        
        return visualizations
    
    def _export_results(self, analysis_results: Dict[str, Any]) -> None:
        """
        Export analysis results in specified formats.
        
        Args:
            analysis_results: Analysis results to export
        """
        for fmt in self.export_formats:
            if fmt == "json":
                output_path = os.path.join(self.output_dir, "analysis_results.json")
                with open(output_path, 'w') as f:
                    json.dump(analysis_results, f, indent=2)
                logger.info(f"Exported analysis results to {output_path}")
            
            elif fmt == "html":
                try:
                    self._export_html(analysis_results)
                except Exception as e:
                    logger.error(f"Error exporting to HTML: {e}")
    
    def _export_html(self, analysis_results: Dict[str, Any]) -> None:
        """
        Export analysis results as HTML report.
        
        Args:
            analysis_results: Analysis results to export
        """
        output_path = os.path.join(self.output_dir, "analysis_report.html")
        
        # Simple HTML template
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Gaelic Football Match Analysis</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2, h3 { color: #2c3e50; }
                .section { margin-bottom: 30px; }
                .recommendation { border-left: 4px solid #3498db; padding-left: 15px; margin-bottom: 15px; }
                .high { border-color: #e74c3c; }
                .medium { border-color: #f39c12; }
                .low { border-color: #2ecc71; }
                .pattern { background-color: #f8f9fa; padding: 10px; margin-bottom: 10px; border-radius: 5px; }
                img { max-width: 100%; height: auto; margin: 10px 0; }
            </style>
        </head>
        <body>
            <h1>Gaelic Football Match Analysis</h1>
            
            <div class="section">
                <h2>Analysis Summary</h2>
                <p>Match analyzed: {source}</p>
                <p>Analysis date: {date}</p>
                <p>Number of plays analyzed: {play_count}</p>
            </div>
            
            <div class="section">
                <h2>Key Recommendations</h2>
                {recommendations}
            </div>
            
            <div class="section">
                <h2>Patterns Identified</h2>
                {patterns}
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
                {visualizations}
            </div>
        </body>
        </html>
        """
        
        # Format recommendations HTML
        recommendations_html = ""
        for rec in analysis_results.get("recommendations", []):
            priority = rec.get("priority", "low")
            recommendations_html += f"""
            <div class="recommendation {priority}">
                <h3>{rec.get('category')}: {rec.get('recommendation')}</h3>
                <p><strong>Based on:</strong> {rec.get('based_on')}</p>
                <p><strong>Priority:</strong> {priority.capitalize()}</p>
            </div>
            """
        
        # Format patterns HTML
        patterns_html = ""
        for pattern in analysis_results.get("patterns", [])[:10]:  # Show top 10 patterns
            patterns_html += f"""
            <div class="pattern">
                <h3>{pattern.get('type', 'Pattern').capitalize()}: {pattern.get('keyword', pattern.get('play_type', ''))}</h3>
                <p>{pattern.get('description', '')}</p>
                <p><strong>Confidence:</strong> {pattern.get('confidence', 0):.2f}</p>
            </div>
            """
        
        # Format visualizations HTML
        visualizations_html = ""
        for name, path in analysis_results.get("visualizations", {}).items():
            # Convert paths to relative for HTML
            rel_path = os.path.basename(path)
            visualizations_html += f"""
            <div class="visualization">
                <h3>{name.replace('_', ' ').title()}</h3>
                <img src="{rel_path}" alt="{name}">
            </div>
            """
        
        # Fill in the template
        html = html.format(
            source=analysis_results.get('metadata', {}).get('source', 'Unknown'),
            date=analysis_results.get('metadata', {}).get('analysis_date', 'Unknown'),
            play_count=analysis_results.get('metadata', {}).get('play_count', 0),
            recommendations=recommendations_html,
            patterns=patterns_html,
            visualizations=visualizations_html
        )
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(html)
            
        logger.info(f"Exported HTML report to {output_path}")
