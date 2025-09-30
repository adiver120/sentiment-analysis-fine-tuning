import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from config import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
    def load_data(self, file_path):
        """Load and prepare the dataset"""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded dataset with {len(df)} samples")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def preprocess_data(self, df):
        """Clean and preprocess the data"""
        # Remove duplicates
        df = df.drop_duplicates(subset=['text'])
        
        # Handle missing values
        df = df.dropna(subset=['text', 'label'])
        
        # Ensure labels are integers
        df['label'] = df['label'].astype(int)
        
        logger.info(f"Data after preprocessing: {len(df)} samples")
        return df
    
    def split_data(self, df):
        """Split data into train, validation, and test sets"""
        train_df, temp_df = train_test_split(
            df, test_size=config.val_split + config.test_split, 
            random_state=42, stratify=df['label']
        )
        
        val_df, test_df = train_test_split(
            temp_df, test_size=config.test_split/(config.val_split + config.test_split), 
            random_state=42, stratify=temp_df['label']
        )
        
        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        return train_df, val_df, test_df
    
    def tokenize_function(self, examples):
        """Tokenize the text data"""
        return self.tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=config.max_length,
            return_tensors=None
        )
    
    def prepare_datasets(self, file_path):
        """Main method to prepare all datasets"""
        # Load and preprocess data
        df = self.load_data(file_path)
        df = self.preprocess_data(df)
        
        # Split data
        train_df, val_df, test_df = self.split_data(df)
        
        # Create Hugging Face datasets
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        test_dataset = Dataset.from_pandas(test_df)
        
        # Tokenize datasets
        train_dataset = train_dataset.map(self.tokenize_function, batched=True)
        val_dataset = val_dataset.map(self.tokenize_function, batched=True)
        test_dataset = test_dataset.map(self.tokenize_function, batched=True)
        
        # Set format for PyTorch
        train_dataset = train_dataset.remove_columns(['__index_level_0__'] if '__index_level_0__' in train_dataset.column_names else [])
        val_dataset = val_dataset.remove_columns(['__index_level_0__'] if '__index_level_0__' in val_dataset.column_names else [])
        test_dataset = test_dataset.remove_columns(['__index_level_0__'] if '__index_level_0__' in test_dataset.column_names else [])
        
        train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })
        
        return dataset_dict

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    datasets = preprocessor.prepare_datasets(config.data_path)
    print("Dataset preparation completed!")