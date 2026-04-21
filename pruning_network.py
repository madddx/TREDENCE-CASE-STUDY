"""
Self-Pruning Neural Network Implementation
Case Study for Tredence Analytics AI Engineering Internship
Author: [Your Name]
Description: A neural network that learns to prune itself during training using gated weights
and L1 sparsity regularization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
from typing import List, Tuple, Dict
import seaborn as sns


# ============================================================================
# PART 1: PRUNABLE LINEAR LAYER
# ============================================================================

class PrunableLinear(nn.Module):
    """
    A custom linear layer with learnable gate parameters for pruning.
    
    Each weight in the layer has an associated gate (between 0 and 1).
    During training, gates that are close to 0 effectively "prune" the weight.
    
    Forward pass:
        gates = sigmoid(gate_scores)
        pruned_weights = weight * gates
        output = pruned_weights @ input + bias
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Initialize the PrunableLinear layer.
        
        Args:
            in_features (int): Number of input features
            out_features (int): Number of output features
            bias (bool): Whether to include bias term
        """
        super(PrunableLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard weight parameter
        self.weight = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        
        # Gate scores: learnable parameters that control which weights are active
        # Same shape as weight
        self.gate_scores = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        
        # Bias parameter
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize weights and gate_scores with proper initialization schemes."""
        # Initialize weights using kaiming uniform (standard for linear layers)
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        
        # Initialize gate_scores: start closer to 1 (all gates active initially)
        # This encourages the network to learn which ones to prune
        nn.init.constant_(self.gate_scores, 0.5)
        
        # Initialize bias
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with gated weights.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features)
        """
        # Apply sigmoid to gate_scores to get values between 0 and 1
        gates = torch.sigmoid(self.gate_scores)
        
        # Element-wise multiplication of weights with gates
        pruned_weights = self.weight * gates
        
        # Standard linear transformation with pruned weights
        return F.linear(x, pruned_weights, self.bias)
    
    def get_sparsity_loss(self) -> torch.Tensor:
        """
        Calculate L1 regularization loss on gate values.
        L1 loss encourages sparsity (pushing values towards 0).
        
        Returns:
            torch.Tensor: Sparsity loss (scalar)
        """
        gates = torch.sigmoid(self.gate_scores)
        return torch.sum(gates)
    
    def get_sparsity_level(self, threshold: float = 1e-2) -> float:
        """
        Calculate the percentage of weights that are effectively pruned.
        
        Args:
            threshold (float): Gate values below this are considered pruned
        
        Returns:
            float: Sparsity percentage (0-100)
        """
        gates = torch.sigmoid(self.gate_scores).detach()
        pruned_count = (gates < threshold).sum().item()
        total_count = gates.numel()
        return (pruned_count / total_count) * 100.0
    
    def extra_repr(self) -> str:
        """String representation of the layer."""
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


# ============================================================================
# PART 2: SELF-PRUNING NEURAL NETWORK
# ============================================================================

class SelfPruningNetwork(nn.Module):
    """
    A simple feed-forward neural network using PrunableLinear layers.
    Designed for CIFAR-10 image classification.
    """
    
    def __init__(self, input_size: int = 3072, hidden_sizes: List[int] = None, 
                 num_classes: int = 10):
        """
        Initialize the network.
        
        Args:
            input_size (int): Input feature size (32x32x3 = 3072 for CIFAR-10)
            hidden_sizes (List[int]): Sizes of hidden layers
            num_classes (int): Number of output classes (10 for CIFAR-10)
        """
        super(SelfPruningNetwork, self).__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [512, 256]
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers with ReLU activations
        for hidden_size in hidden_sizes:
            layers.append(PrunableLinear(prev_size, hidden_size))
            layers.append(nn.ReLU(inplace=True))
            prev_size = hidden_size
        
        # Output layer
        layers.append(PrunableLinear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor (flattened CIFAR-10 images)
        
        Returns:
            torch.Tensor: Logits for each class
        """
        return self.network(x)
    
    def get_total_sparsity_loss(self) -> torch.Tensor:
        """
        Aggregate sparsity loss from all PrunableLinear layers.
        
        Returns:
            torch.Tensor: Sum of sparsity losses from all layers
        """
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                total_loss = total_loss + module.get_sparsity_loss()
        return total_loss
    
    def get_sparsity_levels(self, threshold: float = 1e-2) -> Dict[str, float]:
        """
        Get sparsity levels for each PrunableLinear layer.
        
        Args:
            threshold (float): Gate threshold for pruning
        
        Returns:
            Dict[str, float]: Sparsity levels per layer
        """
        sparsity_dict = {}
        layer_idx = 0
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                sparsity_dict[f'layer_{layer_idx}'] = module.get_sparsity_level(threshold)
                layer_idx += 1
        return sparsity_dict
    
    def get_overall_sparsity(self, threshold: float = 1e-2) -> float:
        """
        Get overall sparsity level across all layers.
        
        Args:
            threshold (float): Gate threshold for pruning
        
        Returns:
            float: Overall sparsity percentage
        """
        total_params = 0
        pruned_params = 0
        
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores).detach()
                pruned_params += (gates < threshold).sum().item()
                total_params += gates.numel()
        
        if total_params == 0:
            return 0.0
        return (pruned_params / total_params) * 100.0
    
    def get_all_gates(self) -> np.ndarray:
        """
        Collect all gate values from the network for visualization.
        
        Returns:
            np.ndarray: Flattened array of all gate values
        """
        all_gates = []
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores).detach().cpu().numpy()
                all_gates.append(gates.flatten())
        return np.concatenate(all_gates)


# ============================================================================
# PART 3: TRAINING AND EVALUATION
# ============================================================================

def load_cifar10_data(batch_size: int = 128) -> Tuple[DataLoader, DataLoader]:
    """
    Load CIFAR-10 dataset with appropriate preprocessing.
    
    Args:
        batch_size (int): Batch size for DataLoader
    
    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, test_loader)
    """
    # Normalization statistics for CIFAR-10
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # No augmentation for testing
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Download and load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return train_loader, test_loader


def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer,
                criterion: nn.Module, lambda_sparsity: float, device: str) -> float:
    """
    Train the model for one epoch.
    
    Args:
        model (nn.Module): The neural network
        train_loader (DataLoader): Training data loader
        optimizer (optim.Optimizer): Optimizer
        criterion (nn.Module): Classification loss function
        lambda_sparsity (float): Weight for sparsity loss
        device (str): Device to train on ('cpu' or 'cuda')
    
    Returns:
        float: Average total loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc='Training', leave=False)
    
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Flatten images for the network
        images = images.view(images.size(0), -1)
        
        # Forward pass
        outputs = model(images)
        
        # Calculate classification loss
        class_loss = criterion(outputs, labels)
        
        # Calculate sparsity loss
        sparsity_loss = model.get_total_sparsity_loss()
        
        # Total loss
        total_loss_batch = class_loss + lambda_sparsity * sparsity_loss
        
        # Backward pass and optimization
        optimizer.zero_grad()
        total_loss_batch.backward()
        optimizer.step()
        
        total_loss += total_loss_batch.item()
        num_batches += 1
        
        progress_bar.set_postfix({'loss': total_loss / num_batches})
    
    return total_loss / num_batches


def evaluate(model: nn.Module, test_loader: DataLoader, device: str) -> Tuple[float, float]:
    """
    Evaluate the model on test set.
    
    Args:
        model (nn.Module): The neural network
        test_loader (DataLoader): Test data loader
        device (str): Device to evaluate on
    
    Returns:
        Tuple[float, float]: (accuracy, average_loss)
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating', leave=False):
            images = images.to(device)
            labels = labels.to(device)
            
            # Flatten images
            images = images.view(images.size(0), -1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()
    
    accuracy = correct / total
    avg_loss = total_loss / len(test_loader)
    
    return accuracy, avg_loss


def train_model(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
                lambda_sparsity: float, num_epochs: int = 30, 
                learning_rate: float = 0.001, device: str = 'cpu') -> Dict:
    """
    Complete training pipeline with validation.
    
    Args:
        model (nn.Module): The neural network
        train_loader (DataLoader): Training data loader
        test_loader (DataLoader): Test data loader
        lambda_sparsity (float): Sparsity regularization weight
        num_epochs (int): Number of training epochs
        learning_rate (float): Learning rate for optimizer
        device (str): Device to train on
    
    Returns:
        Dict: Training history and metrics
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    history = {
        'train_loss': [],
        'test_accuracy': [],
        'test_loss': [],
        'sparsity': []
    }
    
    best_accuracy = 0.0
    
    print(f'\n{"="*70}')
    print(f'Training with λ = {lambda_sparsity}')
    print(f'{"="*70}')
    
    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, 
                                lambda_sparsity, device)
        
        # Evaluate
        test_acc, test_loss = evaluate(model, test_loader, device)
        sparsity = model.get_overall_sparsity()
        
        history['train_loss'].append(train_loss)
        history['test_accuracy'].append(test_acc)
        history['test_loss'].append(test_loss)
        history['sparsity'].append(sparsity)
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model_state = model.state_dict().copy()
        
        scheduler.step()
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | '
                  f'Test Acc: {test_acc:.4f} | Sparsity: {sparsity:.2f}%')
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    final_acc, final_loss = evaluate(model, test_loader, device)
    final_sparsity = model.get_overall_sparsity()
    
    print(f'\n{"="*70}')
    print(f'Final Results (λ = {lambda_sparsity}):')
    print(f'  Test Accuracy: {final_acc:.4f}')
    print(f'  Overall Sparsity: {final_sparsity:.2f}%')
    print(f'{"="*70}\n')
    
    history['final_accuracy'] = final_acc
    history['final_sparsity'] = final_sparsity
    history['lambda'] = lambda_sparsity
    
    return history


def plot_gate_distribution(model: nn.Module, title: str = 'Gate Values Distribution', 
                          save_path: str = 'gate_distribution.png'):
    """
    Plot the distribution of gate values for visualization.
    
    Args:
        model (nn.Module): The trained network
        title (str): Plot title
        save_path (str): Path to save the figure
    """
    gates = model.get_all_gates()
    
    plt.figure(figsize=(10, 6))
    plt.hist(gates, bins=100, edgecolor='black', alpha=0.7, color='steelblue')
    plt.axvline(x=0.01, color='red', linestyle='--', linewidth=2, 
               label='Pruning Threshold (0.01)')
    plt.xlabel('Gate Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Gate distribution plot saved to {save_path}')
    plt.close()


def plot_training_curves(history_list: List[Dict], save_path: str = 'training_curves.png'):
    """
    Plot training curves for different lambda values.
    
    Args:
        history_list (List[Dict]): List of training histories
        save_path (str): Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Curves Across Different λ Values', fontsize=16, fontweight='bold')
    
    for history in history_list:
        lambda_val = history['lambda']
        label = f"λ = {lambda_val}"
        
        # Train loss
        axes[0, 0].plot(history['train_loss'], marker='o', label=label, markersize=3, alpha=0.7)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Training Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Test accuracy
        axes[0, 1].plot(history['test_accuracy'], marker='o', label=label, markersize=3, alpha=0.7)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Test Accuracy')
        axes[0, 1].set_title('Test Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Test loss
        axes[1, 0].plot(history['test_loss'], marker='o', label=label, markersize=3, alpha=0.7)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Test Loss')
        axes[1, 0].set_title('Test Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Sparsity
        axes[1, 1].plot(history['sparsity'], marker='o', label=label, markersize=3, alpha=0.7)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Sparsity (%)')
        axes[1, 1].set_title('Network Sparsity Over Time')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Training curves saved to {save_path}')
    plt.close()


def create_results_table(history_list: List[Dict], save_path: str = 'results_table.txt'):
    """
    Create and save a summary table of results.
    
    Args:
        history_list (List[Dict]): List of training histories
        save_path (str): Path to save the table
    """
    print('\n' + '='*70)
    print('RESULTS SUMMARY TABLE')
    print('='*70)
    print(f'{"Lambda":<12} {"Test Accuracy":<18} {"Sparsity (%)":<15}')
    print('-'*70)
    
    with open(save_path, 'w') as f:
        f.write('='*70 + '\n')
        f.write('RESULTS SUMMARY TABLE\n')
        f.write('='*70 + '\n')
        f.write(f'{"Lambda":<12} {"Test Accuracy":<18} {"Sparsity (%)":<15}\n')
        f.write('-'*70 + '\n')
        
        for history in history_list:
            lambda_val = history['lambda']
            accuracy = history['final_accuracy']
            sparsity = history['final_sparsity']
            
            row = f'{lambda_val:<12.6f} {accuracy:<18.4f} {sparsity:<15.2f}\n'
            print(row, end='')
            f.write(row)
    
    print('='*70 + '\n')
    print(f'Results table saved to {save_path}')


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to run the complete pipeline."""
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Create output directory
    output_dir = Path('./results')
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print('\nLoading CIFAR-10 dataset...')
    train_loader, test_loader = load_cifar10_data(batch_size=128)
    print('Dataset loaded successfully!')
    
    # Define lambda values to test
    lambda_values = [0.0001, 0.001, 0.01]  # Low, Medium, High sparsity
    
    all_histories = []
    best_models = {}
    
    # Train models with different lambda values
    for lambda_val in lambda_values:
        print(f'\n{"#"*70}')
        print(f'Training model with λ = {lambda_val}')
        print(f'{"#"*70}')
        
        # Create fresh model
        model = SelfPruningNetwork(input_size=3072, hidden_sizes=[512, 256], num_classes=10)
        model = model.to(device)
        
        # Train
        history = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            lambda_sparsity=lambda_val,
            num_epochs=30,
            learning_rate=0.001,
            device=device
        )
        
        all_histories.append(history)
        best_models[lambda_val] = model
    
    # Create results table
    create_results_table(all_histories, save_path=str(output_dir / 'results_table.txt'))
    
    # Plot training curves
    plot_training_curves(all_histories, save_path=str(output_dir / 'training_curves.png'))
    
    # Plot gate distributions for each model
    for lambda_val, model in best_models.items():
        plot_gate_distribution(
            model,
            title=f'Gate Values Distribution (λ = {lambda_val})',
            save_path=str(output_dir / f'gate_distribution_lambda_{lambda_val}.png')
        )
    
    # Save detailed results to JSON
    json_results = {
        str(h['lambda']): {
            'final_accuracy': h['final_accuracy'],
            'final_sparsity': h['final_sparsity'],
            'training_history': {
                'train_loss': h['train_loss'],
                'test_accuracy': [float(acc) for acc in h['test_accuracy']],
                'sparsity': h['sparsity']
            }
        }
        for h in all_histories
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f'\nAll results saved to {output_dir}/')


if __name__ == '__main__':
    main()
