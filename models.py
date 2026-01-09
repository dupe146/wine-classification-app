"""
PyTorch implementations of classification models for Wine Dataset
Implements: Logistic Regression, Decision Tree-like Network, and SVM-like Network
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import time


class LogisticRegressionModel(nn.Module):
    """
    PyTorch implementation of Logistic Regression
    Single layer neural network with softmax activation
    """
    def __init__(self, input_dim, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)


class DecisionTreeLikeNN(nn.Module):
    """
    Neural Network mimicking Decision Tree behavior
    Multiple layers with ReLU activations for hierarchical feature learning
    """
    def __init__(self, input_dim, num_classes, hidden_dim=64):
        super(DecisionTreeLikeNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class SVMLikeNN(nn.Module):
    """
    Neural Network with SVM-like behavior using hinge loss
    Deep network with batch normalization
    """
    def __init__(self, input_dim, num_classes, hidden_dim=128):
        super(SVMLikeNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


class WineClassifier:
    """
    Wrapper class for training and evaluating PyTorch models
    """
    def __init__(self, model, model_name, device='cpu'):
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.training_history = {'loss': [], 'accuracy': []}
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=100, batch_size=16, learning_rate=0.01, 
              optimizer_type='adam', verbose=True):
        """
        Train the model with specified hyperparameters
        """
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Optimizer selection
        if optimizer_type.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_type.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer_type.lower() == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=learning_rate)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        start_time = time.time()
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
                _, predicted = torch.max(val_outputs, 1)
                val_accuracy = (predicted == y_val_tensor).sum().item() / len(y_val_tensor)
            
            self.training_history['loss'].append(epoch_loss / len(train_loader))
            self.training_history['accuracy'].append(val_accuracy)
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], "
                      f"Loss: {epoch_loss/len(train_loader):.4f}, "
                      f"Val Accuracy: {val_accuracy:.4f}")
        
        training_time = time.time() - start_time
        if verbose:
            print(f"\nTraining completed in {training_time:.2f} seconds")
        
        return training_time
    
    def predict(self, X_test):
        """
        Make predictions on test data
        """
        self.model.eval()
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_test_tensor)
            _, predicted = torch.max(outputs, 1)
        
        return predicted.cpu().numpy()
    
    def evaluate(self, X_test, y_test, target_names=None):
        """
        Evaluate model performance
        """
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        conf_matrix = confusion_matrix(y_test, predictions)
        class_report = classification_report(y_test, predictions, 
                                             target_names=target_names,
                                             output_dict=True)
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'predictions': predictions
        }
    
    def save_model(self, filepath):
        """
        Save model for future deployment
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'training_history': self.training_history
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load pre-trained model
        """
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model_name = checkpoint['model_name']
        self.training_history = checkpoint['training_history']
        print(f"Model loaded from {filepath}")