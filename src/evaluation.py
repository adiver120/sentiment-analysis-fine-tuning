import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from config import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model_path, datasets):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.datasets = datasets
        
        # Label mapping
        self.label_names = {0: "Negative", 1: "Neutral", 2: "Positive"}
    
    def evaluate_model(self):
        """Comprehensive model evaluation"""
        test_dataloader = DataLoader(
            self.datasets['test'],
            batch_size=config.batch_size,
        )
        
        # Get predictions
        all_predictions, all_labels, all_probabilities = self.get_predictions(test_dataloader)
        
        # Generate reports
        report_df = self.generate_classification_report(all_predictions, all_labels)
        self.plot_confusion_matrix(all_predictions, all_labels)
        self.calculate_additional_metrics(all_predictions, all_labels)
        
        return all_predictions, all_labels, all_probabilities
    
    def get_predictions(self, dataloader):
        """Get model predictions - FIXED VERSION"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        for batch in dataloader:
            # Separate inputs and labels - FIXED
            inputs = {
                'input_ids': batch['input_ids'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device)
            }
            labels = batch['label'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)  # Don't pass labels during inference
            
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
        
        return all_predictions, all_labels, all_probabilities
    
    def generate_classification_report(self, predictions, labels):
        """Generate and display classification report"""
        report = classification_report(
            labels, 
            predictions, 
            target_names=[self.label_names[i] for i in range(config.num_labels)],
            output_dict=True
        )
        
        print("\n" + "="*50)
        print("CLASSIFICATION REPORT")
        print("="*50)
        print(classification_report(
            labels, 
            predictions, 
            target_names=[self.label_names[i] for i in range(config.num_labels)]
        ))
        
        # Convert to DataFrame for better visualization
        report_df = pd.DataFrame(report).transpose()
        return report_df
    
    def plot_confusion_matrix(self, predictions, labels):
        """Plot confusion matrix"""
        cm = confusion_matrix(labels, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=[self.label_names[i] for i in range(config.num_labels)],
                   yticklabels=[self.label_names[i] for i in range(config.num_labels)])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def calculate_additional_metrics(self, predictions, labels):
        """Calculate additional performance metrics"""
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        print("\n" + "="*50)
        print("ADDITIONAL METRICS")
        print("="*50)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Weighted Precision: {precision:.4f}")
        print(f"Weighted Recall: {recall:.4f}")
        print(f"Weighted F1-Score: {f1:.4f}")
        
        # Per-class metrics
        print("\nPER-CLASS METRICS:")
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            labels, predictions, average=None
        )
        
        for i in range(config.num_labels):
            print(f"{self.label_names[i]}: "
                  f"Precision={precision_per_class[i]:.4f}, "
                  f"Recall={recall_per_class[i]:.4f}, "
                  f"F1={f1_per_class[i]:.4f}, "
                  f"Support={support_per_class[i]}")

if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor
    
    # Load data and evaluate
    preprocessor = DataPreprocessor()
    datasets = preprocessor.prepare_datasets(config.data_path)
    
    evaluator = ModelEvaluator(config.model_save_path, datasets)
    predictions, labels, probabilities = evaluator.evaluate_model()