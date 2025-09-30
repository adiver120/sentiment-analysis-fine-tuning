#!/usr/bin/env python3
"""
Script to download and setup the pre-trained model for fine-tuning
"""

import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_and_setup_model():
    """Download the pre-trained model and tokenizer"""
    
    # Create models directory if it doesn't exist
    os.makedirs("models/saved_models", exist_ok=True)
    
    model_path = "models/pretrained_model"
    
    try:
        logger.info(f"Downloading {config.model_name} model and tokenizer...")
        
        # Download model
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels
        )
        
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # Save locally
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        
        logger.info(f"Model and tokenizer saved to {model_path}")
        
        # Test the model
        logger.info("Testing the downloaded model...")
        test_model_load(model_path)
        
        return model_path
        
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        raise

def test_model_load(model_path):
    """Test that the model can be loaded correctly"""
    try:
        # Test loading
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Test inference
        test_text = "This is a test sentence."
        inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logger.info("Model loaded and tested successfully!")
        logger.info(f"Model architecture: {type(model)}")
        logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    except Exception as e:
        logger.error(f"Error testing model: {e}")
        raise

def download_alternative_models():
    """Download alternative models for comparison"""
    models_to_download = [
        "distilbert-base-uncased",
        "bert-base-uncased", 
        "roberta-base",
        "cardiffnlp/twitter-roberta-base-sentiment-latest"
    ]
    
    for model_name in models_to_download:
        try:
            logger.info(f"Downloading {model_name}...")
            model_path = f"models/{model_name.replace('/', '_')}"
            
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=config.num_labels
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            
            logger.info(f"✓ {model_name} downloaded successfully")
            
        except Exception as e:
            logger.warning(f"✗ Failed to download {model_name}: {e}")

if __name__ == "__main__":
    print("Model Download Script")
    print("=" * 50)
    
    # Download main model
    model_path = download_and_setup_model()
    
    # Uncomment to download alternative models
    # download_alternative_models()
    
    print("\nModel setup completed! You can now run the training.")