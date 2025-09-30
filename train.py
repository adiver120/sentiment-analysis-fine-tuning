#!/usr/bin/env python3

import logging
from src.data_preprocessing import DataPreprocessor
from src.model_training import SentimentTrainer
from src.evaluation import ModelEvaluator
from config import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Starting sentiment analysis fine-tuning...")
        
        # Step 1: Prepare data
        logger.info("Step 1: Preparing data...")
        preprocessor = DataPreprocessor()
        datasets = preprocessor.prepare_datasets(config.data_path)
        
        # Step 2: Train model
        logger.info("Step 2: Training model...")
        trainer = SentimentTrainer(datasets)
        trainer.train()
        trainer.save_model()
        
        # Step 3: Evaluate model
        logger.info("Step 3: Evaluating model...")
        evaluator = ModelEvaluator(config.model_save_path, datasets)
        predictions, labels, probabilities = evaluator.evaluate_model()
        
        logger.info("Fine-tuning completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()