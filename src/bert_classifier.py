"""
DistilBERT Comment Classifier
Fine-tune DistilBERT for comment categorization with improved performance
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import pickle
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Category mapping
CATEGORIES = [
    'Constructive Criticism',
    'Emotional',
    'Hate/Abuse',
    'Irrelevant/Spam',
    'Praise',
    'Question/Suggestion',
    'Support',
    'Threat'
]

class CommentDataset(Dataset):
    """Custom Dataset for comment classification"""
    
    def __init__(self, comments, labels, tokenizer, max_length=128):
        self.comments = comments
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.comments)
    
    def __getitem__(self, idx):
        comment = str(self.comments[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            comment,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class DistilBERTClassifier:
    """DistilBERT-based comment classifier"""
    
    def __init__(self, num_labels=8, max_length=128):
        self.num_labels = num_labels
        self.max_length = max_length
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = None
        self.label_encoder = {}
        self.label_decoder = {}
    
    def prepare_data(self, df, test_size=0.2):
        """Prepare data for training"""
        print("Preparing data...")
        
        # Encode labels
        unique_labels = sorted(df['category'].unique())
        self.label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
        self.label_decoder = {idx: label for label, idx in self.label_encoder.items()}
        
        # Convert labels to integers
        df['label_id'] = df['category'].map(self.label_encoder)
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df['comment'].values,
            df['label_id'].values,
            test_size=test_size,
            random_state=42,
            stratify=df['label_id'].values
        )
        
        print(f"Training samples: {len(train_texts):,}")
        print(f"Validation samples: {len(val_texts):,}")
        
        return train_texts, val_texts, train_labels, val_labels
    
    def create_data_loaders(self, train_texts, val_texts, train_labels, val_labels, batch_size=16):
        """Create PyTorch DataLoaders"""
        train_dataset = CommentDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        val_dataset = CommentDataset(val_texts, val_labels, self.tokenizer, self.max_length)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        return train_loader, val_loader
    
    def train(self, df, epochs=3, batch_size=16, learning_rate=2e-5, sample_size=None):
        """Train the DistilBERT model"""
        
        # Sample data if specified (for faster training)
        if sample_size and len(df) > sample_size:
            print(f"Sampling {sample_size:,} comments from {len(df):,} for faster training...")
            df = df.sample(n=sample_size, random_state=42)
        
        # Prepare data
        train_texts, val_texts, train_labels, val_labels = self.prepare_data(df)
        train_loader, val_loader = self.create_data_loaders(
            train_texts, val_texts, train_labels, val_labels, batch_size
        )
        
        # Initialize model
        print("\nInitializing DistilBERT model...")
        self.model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=self.num_labels
        )
        self.model.to(device)
        
        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Training loop
        print(f"\nTraining for {epochs} epochs...")
        print("=" * 60)
        
        best_val_accuracy = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 60)
            
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            progress_bar = tqdm(train_loader, desc="Training")
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                optimizer.zero_grad()
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                train_correct += (predictions == labels).sum().item()
                train_total += labels.size(0)
                
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{train_correct/train_total:.4f}'
                })
            
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = train_correct / train_total
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            all_predictions = []
            all_labels = []
            
            with torch.no_grad():
                progress_bar = tqdm(val_loader, desc="Validation")
                for batch in progress_bar:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    logits = outputs.logits
                    
                    val_loss += loss.item()
                    predictions = torch.argmax(logits, dim=1)
                    val_correct += (predictions == labels).sum().item()
                    val_total += labels.size(0)
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{val_correct/val_total:.4f}'
                    })
            
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = val_correct / val_total
            
            print(f"\nEpoch {epoch + 1} Results:")
            print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                print(f"  ✓ New best validation accuracy: {val_accuracy:.4f}")
        
        # Final evaluation
        print("\n" + "=" * 60)
        print("Final Validation Results:")
        print("=" * 60)
        print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")
        
        # Classification report
        category_names = [self.label_decoder[i] for i in range(self.num_labels)]
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions, target_names=category_names))
        
        return best_val_accuracy
    
    def predict(self, comments, batch_size=16):
        """Predict categories for new comments"""
        if self.model is None:
            raise ValueError("Model not trained or loaded!")
        
        self.model.eval()
        
        if isinstance(comments, str):
            comments = [comments]
        
        # Create dataset
        dummy_labels = [0] * len(comments)
        dataset = CommentDataset(comments, dummy_labels, self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Decode predictions
        predicted_categories = [self.label_decoder[pred] for pred in all_predictions]
        confidences = [max(prob) for prob in all_probabilities]
        
        return predicted_categories, confidences
    
    def save_model(self, model_dir='../models/bert'):
        """Save the trained model"""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)
        
        # Save label encoders
        with open(os.path.join(model_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump({'encoder': self.label_encoder, 'decoder': self.label_decoder}, f)
        
        print(f"\nModel saved to {model_dir}")
    
    def load_model(self, model_dir='../models/bert'):
        """Load a trained model"""
        self.model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
        self.model.to(device)
        
        # Load label encoders
        with open(os.path.join(model_dir, 'label_encoder.pkl'), 'rb') as f:
            encoders = pickle.load(f)
            self.label_encoder = encoders['encoder']
            self.label_decoder = encoders['decoder']
        
        print(f"Model loaded from {model_dir}")


def main():
    """Main training function"""
    print("=" * 60)
    print("DistilBERT Comment Classifier Training")
    print("=" * 60)
    
    # Load dataset
    print("\nLoading dataset...")
    df = pd.read_csv('../data/comments_dataset.csv')
    print(f"Total comments: {len(df):,}")
    print(f"Categories: {df['category'].nunique()}")
    
    # Initialize classifier
    classifier = DistilBERTClassifier(num_labels=len(df['category'].unique()))
    
    # Train model (using a sample for faster training - remove sample_size for full training)
    # For full 116K dataset, this would take hours. Using 10K sample for demonstration.
    accuracy = classifier.train(
        df,
        epochs=3,
        batch_size=16,
        learning_rate=2e-5,
        sample_size=10000  # Remove this line to train on full dataset
    )
    
    # Save model
    classifier.save_model()
    
    # Test predictions
    print("\n" + "=" * 60)
    print("Testing predictions:")
    print("=" * 60)
    
    test_comments = [
        "This is amazing! Great work!",
        "This is terrible. You should quit.",
        "Good effort but the audio could be better.",
        "Follow me for more content!",
        "Can you make a tutorial on this?"
    ]
    
    predictions, confidences = classifier.predict(test_comments)
    
    for comment, category, confidence in zip(test_comments, predictions, confidences):
        print(f"\nComment: {comment}")
        print(f"Category: {category}")
        print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
