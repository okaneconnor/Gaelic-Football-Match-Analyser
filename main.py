#!/usr/bin/env python3
"""
Gaelic Football Match Analyser - Main Entry Point
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from src.data_processing.video_processor import VideoProcessor
from src.data_processing.data_transformer import DataTransformer
from src.model.model_selector import ModelSelector
from src.model.fine_tuner import FineTuner
from src.inference.inference_pipeline import InferencePipeline
from src.inference.analysis import Analyzer
from src.ui.app import launch_ui

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Gaelic Football Match Analyser'
    )
    
    # Main operation mode
    parser.add_argument('--ui', action='store_true', 
                       help='Launch the user interface (default)')
    
    # Input/output options
    parser.add_argument('--video_path', type=str,
                       help='Path to the input video file')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Directory to save output files')
    
    # Model options
    parser.add_argument('--model', type=str, default='mistral-7b',
                       choices=['llama-2-7b', 'mistral-7b', 'phi-2', 'gpt-j-6b'],
                       help='LLM to use for analysis')
    parser.add_argument('--model_path', type=str,
                       help='Path to a custom model or fine-tuned model')
    
    # Processing options
    parser.add_argument('--skip_video_processing', action='store_true',
                       help='Skip video processing, use existing data')
    parser.add_argument('--skip_fine_tuning', action='store_true',
                       help='Skip model fine-tuning, use base model')
    
    # Debug options
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Create output directory if it doesn't exist
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Output directory: {args.output_dir}")
    
    # Launch UI if no specific command is given or UI flag is set
    if args.ui or (not args.video_path and not args.model_path):
        logger.info("Launching user interface...")
        launch_ui()
        return
    
    # Initialize the pipeline components
    logger.info(f"Initializing with model: {args.model}")
    model_selector = ModelSelector(model_name=args.model, model_path=args.model_path)
    
    # Process video if provided and not skipped
    if args.video_path and not args.skip_video_processing:
        logger.info(f"Processing video: {args.video_path}")
        video_processor = VideoProcessor()
        processed_data = video_processor.process(args.video_path, output_dir=args.output_dir)
        
        logger.info("Transforming data...")
        data_transformer = DataTransformer()
        transformed_data = data_transformer.transform(processed_data)
    else:
        logger.info("Skipping video processing")
        transformed_data = None
    
    # Fine-tune model if not skipped
    if not args.skip_fine_tuning:
        logger.info("Fine-tuning model...")
        fine_tuner = FineTuner(model_selector.get_model(), output_dir=args.output_dir)
        fine_tuned_model = fine_tuner.fine_tune(transformed_data)
    else:
        logger.info("Skipping fine-tuning")
        fine_tuned_model = model_selector.get_model()
    
    # Run inference and analysis
    logger.info("Running inference pipeline...")
    inference_pipeline = InferencePipeline(fine_tuned_model)
    results = inference_pipeline.run(transformed_data or args.video_path)
    
    logger.info("Analyzing results...")
    analyzer = Analyzer()
    analysis_results = analyzer.analyze(results)
    
    # Save results
    logger.info(f"Saving results to {args.output_dir}")
    results_file = Path(args.output_dir) / "analysis_results.json"
    with open(results_file, 'w') as f:
        import json
        json.dump(analysis_results, f, indent=2)
    
    logger.info(f"Analysis complete. Results saved to {results_file}")

if __name__ == "__main__":
    main()
