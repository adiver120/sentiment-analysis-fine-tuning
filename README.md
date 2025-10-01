# Sentiment Analysis Fine-Tuning

A complete Python project for fine-tuning transformer models for sentiment analysis tasks. This project provides an end-to-end pipeline for training, evaluating, and deploying custom sentiment analysis models.

## 🚀 Features

- **Model Fine-tuning**: Fine-tune DistilBERT, BERT, or RoBERTa for sentiment analysis
- **Complete Pipeline**: Data preprocessing, training, evaluation, and inference
- **Visualization**: Comprehensive metrics and confusion matrices
- **Production Ready**: Easy deployment with inference scripts
- **Customizable**: Configurable training parameters and model architectures
- **Interactive**: Jupyter notebooks for exploration and analysis

## 📁 Project Structure

sentiment-analysis-fine-tuning/
├── .gitignore # Git ignore rules
├── README.md # Project documentation
├── requirements.txt # Python dependencies
├── config.py # Configuration settings
├── train.py # Main training script
├── predict.py # Inference script
├── setup.py # Project setup verification
│
├── src/ # Source code
│ ├── init.py
│ ├── data_preprocessing.py # Data loading and preprocessing
│ ├── model_training.py # Model training logic
│ ├── evaluation.py # Model evaluation and metrics
│ ├── inference.py # Prediction interface
│ └── download_model.py # Pre-trained model downloader
│
├── notebooks/ # Jupyter notebooks
│ └── sentiment_analysis_fine_tuning.ipynb # Complete guide
│
├── data/ # Dataset directory
│ ├── raw_data.csv # Sample dataset
│ └── expanded_data.csv # Expanded dataset (auto-generated)
│
└── models/ # Model storage (not pushed to git)
├── pretrained_model/ # Downloaded base models
├── saved_models/ # Fine-tuned models
└── deployment_package/ # Production deployment files