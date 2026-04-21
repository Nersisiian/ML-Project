import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class PricePredictorNN(nn.Module):
    """Neural network for price prediction"""
    
    def __init__(self, input_dim: int, hidden_dims: list = [256, 128, 64]):
        super(PricePredictorNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class NeuralNetworkModel:
    """PyTorch neural network wrapper"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32
    ):
        """Train neural network"""
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        input_dim = X_train.shape[1]
        self.model = PricePredictorNN(
            input_dim=input_dim,
            hidden_dims=self.config.get('hidden_dims', [256, 128, 64])
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 0.001),
            weight_decay=self.config.get('weight_decay', 1e-5)
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=10, factor=0.5
        )
        
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            
            # Validation
            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    X_val_tensor = torch.FloatTensor(X_val).to(self.device)
                    y_val_tensor = torch.FloatTensor(y_val).to(self.device)
                    val_outputs = self.model(X_val_tensor).squeeze()
                    val_loss = criterion(val_outputs, y_val_tensor)
                
                scheduler.step(val_loss)
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss: {avg_loss:.4f}")
        
        logger.info("Training complete")
        
        return self.model
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy().squeeze()
        
        return predictions
    
    def save(self, path: str):
        """Save model to disk"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk"""
        checkpoint = torch.load(path, map_location=self.device)
        input_dim = 50  # Should be saved in config
        self.model = PricePredictorNN(input_dim=input_dim).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint['config']
        logger.info(f"Model loaded from {path}")