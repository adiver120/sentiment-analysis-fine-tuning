#!/usr/bin/env python3

import argparse
from src.inference import SentimentPredictor
from config import config

def main():
    parser = argparse.ArgumentParser(description='Predict sentiment of text')
    parser.add_argument('--text', type=str, help='Text to analyze')
    parser.add_argument('--file', type=str, help='File containing texts to analyze (one per line)')
    
    args = parser.parse_args()
    
    predictor = SentimentPredictor(config.model_save_path)
    
    if args.text:
        # Single text prediction
        result = predictor.predict_sentiment(args.text)
        predictor.print_prediction(result)
    
    elif args.file:
        # Batch prediction from file
        with open(args.file, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        results = predictor.predict_batch(texts)
        
        print("BATCH PREDICTION RESULTS")
        print("=" * 50)
        for result in results:
            predictor.print_prediction(result)
    
    else:
        # Interactive mode
        print("Interactive Sentiment Analysis")
        print("Enter texts to analyze (type 'quit' to exit):")
        print("-" * 50)
        
        while True:
            text = input("\nEnter text: ").strip()
            if text.lower() in ['quit', 'exit', 'q']:
                break
            if text:
                result = predictor.predict_sentiment(text)
                predictor.print_prediction(result)

if __name__ == "__main__":
    main()