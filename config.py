import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    # Model settings
    model_name: str = "distilbert-base-uncased"
    pretrained_model_path: str = "models/pretrained_model"
    num_labels: int = 3
    
    # Training settings
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 4
    max_length: int = 256
    warmup_steps: int = 500
    weight_decay: float = 0.01
    
    # Data settings
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Paths
    data_path: str = "data/raw_data.csv"
    model_save_path: str = "models/saved_models/fine_tuned_sentiment"
    logs_path: str = "logs/"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model alternatives
    alternative_models: dict = None
    
    def __post_init__(self):
        self.alternative_models = {
            "distilbert": "distilbert-base-uncased",
            "bert": "bert-base-uncased",
            "roberta": "roberta-base",
            "twitter_roberta": "cardiffnlp/twitter-roberta-base-sentiment-latest"
        }

config = TrainingConfig()