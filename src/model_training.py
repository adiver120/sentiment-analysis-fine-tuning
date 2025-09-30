import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler
)
from torch.optim import AdamW
from datasets import DatasetDict
import evaluate
import numpy as np
from tqdm.auto import tqdm
from config import config
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentTrainer:
    def __init__(self, datasets):
        self.datasets = datasets
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name, 
            num_labels=config.num_labels
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Metrics
        self.accuracy_metric = evaluate.load("accuracy")
        self.f1_metric = evaluate.load("f1")
        
    def create_data_loaders(self):
        """Create data loaders for training and validation"""
        train_dataloader = DataLoader(
            self.datasets['train'],
            shuffle=True,
            batch_size=config.batch_size,
        )
        
        eval_dataloader = DataLoader(
            self.datasets['validation'],
            batch_size=config.batch_size,
        )
        
        test_dataloader = DataLoader(
            self.datasets['test'],
            batch_size=config.batch_size,
        )
        
        return train_dataloader, eval_dataloader, test_dataloader
    
    def setup_training(self, train_dataloader):
        """Setup optimizer and scheduler"""
        optimizer = AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        
        num_training_steps = config.num_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=num_training_steps
        )
        
        return optimizer, lr_scheduler, num_training_steps
    
    def compute_metrics(self, predictions, labels):
        """Compute metrics for evaluation"""
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = self.accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
        f1 = self.f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"]
        
        return {"accuracy": accuracy, "f1": f1}
    
    def train(self):
        """Main training loop"""
        train_dataloader, eval_dataloader, test_dataloader = self.create_data_loaders()
        optimizer, lr_scheduler, num_training_steps = self.setup_training(train_dataloader)
        
        progress_bar = tqdm(range(num_training_steps))
        
        # Training history
        training_history = {
            'train_loss': [],
            'val_accuracy': [],
            'val_f1': []
        }
        
        self.model.train()
        for epoch in range(config.num_epochs):
            total_loss = 0
            for batch in train_dataloader:
                # Separate inputs and labels - FIXED
                inputs = {
                    'input_ids': batch['input_ids'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device)
                }
                labels = batch['label'].to(self.device)
                
                # Forward pass - CORRECT WAY
                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                
                # Backward pass
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
            
            avg_loss = total_loss / len(train_dataloader)
            eval_results = self.evaluate(eval_dataloader)
            
            # Store history
            training_history['train_loss'].append(avg_loss)
            training_history['val_accuracy'].append(eval_results['accuracy'])
            training_history['val_f1'].append(eval_results['f1'])
            
            logger.info(f"Epoch {epoch + 1}/{config.num_epochs}")
            logger.info(f"Average Loss: {avg_loss:.4f}")
            logger.info(f"Validation Accuracy: {eval_results['accuracy']:.4f}")
            logger.info(f"Validation F1: {eval_results['f1']:.4f}")
            logger.info("-" * 50)
        
        # Final evaluation on test set
        logger.info("Final evaluation on test set:")
        test_results = self.evaluate(test_dataloader)
        logger.info(f"Test Accuracy: {test_results['accuracy']:.4f}")
        logger.info(f"Test F1: {test_results['f1']:.4f}")
        
        return training_history
    
    def evaluate(self, eval_dataloader):
        """Evaluate the model"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0
        
        for batch in eval_dataloader:
            # Separate inputs and labels - FIXED
            inputs = {
                'input_ids': batch['input_ids'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device)
            }
            labels = batch['label'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, labels=labels)
            
            predictions = torch.argmax(outputs.logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_loss += outputs.loss.item()
        
        # Convert to numpy arrays for metric computation
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Compute metrics
        accuracy = self.accuracy_metric.compute(predictions=all_predictions, references=all_labels)["accuracy"]
        f1 = self.f1_metric.compute(predictions=all_predictions, references=all_labels, average="weighted")["f1"]
        
        avg_loss = total_loss / len(eval_dataloader)
        
        results = {
            "accuracy": accuracy,
            "f1": f1,
            "loss": avg_loss
        }
        
        self.model.train()
        return results
    
    def save_model(self):
        """Save the trained model"""
        os.makedirs(config.model_save_path, exist_ok=True)
        self.model.save_pretrained(config.model_save_path)
        self.tokenizer.save_pretrained(config.model_save_path)
        logger.info(f"Model saved to {config.model_save_path}")

if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor
    
    # Prepare data and train
    preprocessor = DataPreprocessor()
    datasets = preprocessor.prepare_datasets(config.data_path)
    
    trainer = SentimentTrainer(datasets)
    trainer.train()
    trainer.save_model()