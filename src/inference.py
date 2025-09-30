import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from config import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentPredictor:
    def __init__(self, model_path):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.model.eval()
        
        self.label_names = {0: "Negative", 1: "Neutral", 2: "Positive"}
    
    def predict_sentiment(self, text):
        """Predict sentiment for a single text"""
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=config.max_length,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(outputs.logits, dim=-1)
        
        confidence = probabilities[0][prediction].item()
        label_name = self.label_names[prediction.item()]
        
        return {
            'text': text,
            'prediction': label_name,
            'confidence': confidence,
            'probabilities': {
                self.label_names[i]: prob.item() 
                for i, prob in enumerate(probabilities[0])
            }
        }
    
    def predict_batch(self, texts):
        """Predict sentiment for multiple texts"""
        results = []
        for text in texts:
            result = self.predict_sentiment(text)
            results.append(result)
        return results
    
    def print_prediction(self, result):
        """Print prediction results in a formatted way"""
        print(f"\nText: {result['text']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("Probabilities:")
        for label, prob in result['probabilities'].items():
            print(f"  {label}: {prob:.4f}")

if __name__ == "__main__":
    # Example usage
    predictor = SentimentPredictor(config.model_save_path)
    
    test_texts = [
        "I love this product! It's absolutely amazing!",
        "This is terrible and I hate it.",
        "It's okay, nothing special.",
        "The service was average but the food was great."
    ]
    
    print("SENTIMENT ANALYSIS PREDICTIONS")
    print("=" * 50)
    
    for text in test_texts:
        result = predictor.predict_sentiment(text)
        predictor.print_prediction(result)