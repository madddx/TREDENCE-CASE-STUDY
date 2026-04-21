# Self-Pruning Neural Network
**Case Study Submission for Tredence Analytics AI Engineering Internship 2025**

> A neural network that learns to prune itself during training using gated weights and L1 sparsity regularization.

## 🎯 Project Overview

This project implements a **self-pruning neural network** that dynamically removes unimportant weights during training rather than as a post-training step. The network learns which connections matter by:

1. **Associating learnable "gates"** with each weight (values between 0-1)
2. **Scaling weights** by their corresponding gates during forward passes
3. **Using L1 regularization** on gate values to encourage sparsity
4. **Adapting network architecture** on-the-fly during training

The result is a flexible approach to model compression that balances accuracy and efficiency.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/self-pruning-network.git
cd self-pruning-network

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Code

```bash
# Run complete pipeline with default parameters
python pruning_network.py

# The script will:
# 1. Download CIFAR-10 dataset
# 2. Train models with 3 different λ values
# 3. Generate visualizations and results table
# 4. Save everything to results/ directory
```

## 📊 Project Structure

```
self-pruning-network/
├── pruning_network.py           # Main implementation (single file, easy to run)
├── REPORT.md                    # Detailed technical report
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── .gitignore                   # Git ignore rules
└── results/                     # Generated outputs
    ├── results_table.txt        # Summary results
    ├── results.json             # Detailed metrics (JSON)
    ├── training_curves.png      # 4-panel training visualization
    └── gate_distribution_*.png  # Gate value histograms
```

## 🧠 Core Components

### 1. PrunableLinear Layer
A custom linear layer with learnable gates:

```python
class PrunableLinear(nn.Module):
    """
    Linear layer with gated weights for pruning.
    
    Forward: output = (weight * sigmoid(gate_scores)) @ input + bias
    """
```

**Key methods:**
- `forward()`: Applies gated weights
- `get_sparsity_loss()`: Returns L1 loss on gates
- `get_sparsity_level()`: Computes percentage of pruned weights

### 2. SelfPruningNetwork
Feed-forward network using PrunableLinear layers:

```
Input → PrunableLinear → ReLU → 
         PrunableLinear → ReLU → 
         PrunableLinear → Output
```

### 3. Training Loop
Complete training pipeline with:
- Classification loss (CrossEntropyLoss)
- Sparsity loss (L1 norm of gates)
- Adam optimizer with cosine annealing
- Best model selection and evaluation

## 📈 Results Interpretation

### Expected Outputs

A successful run produces:

1. **Results Table**: Summary of accuracy vs. sparsity trade-off

| Lambda | Test Accuracy | Sparsity (%) |
|--------|---------------|--------------|
| 0.0001 | ~0.58         | ~12%         |
| 0.001  | ~0.56         | ~29%         |
| 0.01   | ~0.51         | ~46%         |

2. **Gate Distribution Plots**: Histograms showing bimodal distribution
   - Spike at 0: Pruned weights
   - Cluster away from 0: Active weights

3. **Training Curves**: 4-panel visualization
   - Training loss convergence
   - Test accuracy progression
   - Test loss trend
   - Sparsity increase over epochs

### Key Metrics

- **Sparsity**: % of weights with gate value < 0.01
- **Accuracy**: Classification accuracy on CIFAR-10 test set
- **Trade-off**: Higher λ → more sparsity → lower accuracy

## 🔍 Understanding the Code

### Loss Function

```python
Total Loss = ClassificationLoss + λ × SparsityLoss
           = CrossEntropyLoss + λ × Σ(sigmoid(gate_scores))
```

Where:
- **ClassificationLoss**: Standard cross-entropy for classification
- **SparsityLoss**: L1 norm of sigmoid gates (encourages zeros)
- **λ (lambda)**: Hyperparameter controlling pruning aggressiveness

### Gate Mechanism

For each weight in a layer:
```
gate = sigmoid(gate_score)      # Value in [0, 1]
pruned_weight = weight × gate   # 0 → weight off, 1 → weight on
```

### Why L1 Regularization?

L1 regularization (sum of absolute values) naturally drives parameters to exactly 0:
- Unlike L2 (shrinks values), L1 can completely eliminate weights
- Creates sparse solutions (many exact zeros)
- Well-suited for sparsity-inducing regularization

## 🎛️ Customization

### Modify Network Architecture

```python
# In main(), change hidden layer sizes:
model = SelfPruningNetwork(
    input_size=3072,
    hidden_sizes=[1024, 512, 256],  # Deeper network
    num_classes=10
)
```

### Adjust Training Parameters

```python
history = train_model(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    lambda_sparsity=0.005,      # Change sparsity weight
    num_epochs=50,              # More epochs
    learning_rate=0.0005,       # Lower learning rate
    device=device
)
```

### Test Different Lambda Values

```python
# In main(), modify:
lambda_values = [0.00001, 0.0001, 0.001, 0.01, 0.1]
```

## 📋 Requirements

```
torch>=2.0.0           # Deep learning framework
torchvision>=0.15.0    # Computer vision datasets
numpy>=1.24.0          # Numerical computing
matplotlib>=3.7.0      # Visualization
seaborn>=0.12.0        # Statistical plotting
tqdm>=4.65.0           # Progress bars
```

Install all at once:
```bash
pip install -r requirements.txt
```

## 🎓 Learning Outcomes

From implementing this project, you'll understand:

✅ How to build custom PyTorch layers with custom backward passes  
✅ Gradient flow through custom operations  
✅ L1 regularization for sparsity induction  
✅ Trade-offs between model size and accuracy  
✅ Complete ML pipeline (data loading → training → evaluation)  
✅ Visualization and analysis of neural networks  
✅ Production-quality Python code practices  

## 📚 Technical Details

For deep dive into methodology, see **[REPORT.md](REPORT.md)** which includes:
- Mathematical formulation
- Detailed architectural decisions
- Why L1 sparsity works
- Experimental results analysis
- Comparison with other pruning methods
- Future extensions

## 🔗 Key Insights

### Why This Works
1. **Learnable gates**: Network learns what to prune, not arbitrary rules
2. **Differentiable**: Entire process is backprop-compatible
3. **Interpretable**: Gate values clearly show importance
4. **Flexible**: Easy to adjust pruning aggressiveness via λ
5. **End-to-end**: No need for post-training pruning steps

### Advantages Over Alternatives
- **vs. Magnitude pruning**: More adaptive, learned during training
- **vs. Binary masks**: Smooth gradients, easier optimization
- **vs. Lottery tickets**: Direct pruning, no retraining needed

## 🚨 Troubleshooting

### Out of Memory
```python
# Reduce batch size
train_loader = DataLoader(train_dataset, batch_size=64, ...)

# Or reduce hidden layer sizes
model = SelfPruningNetwork(hidden_sizes=[256, 128])
```

### Slow Training
- Reduce `num_epochs`
- Reduce model size
- Use smaller dataset subset for testing
- Enable GPU: `device = 'cuda'`

### Low Sparsity with High Lambda
- Increase `num_epochs` (more time to prune)
- Start with smaller `lambda_values`
- Check if gates are updating (add print statements)

## 📧 Contact & Support

For questions or issues:
1. Check **[REPORT.md](REPORT.md)** for technical details
2. Review code comments in **[pruning_network.py](pruning_network.py)**
3. Run with fresh CIFAR-10 download if data issues

## 📜 License

This project is submitted as a case study for Tredence Analytics AI Engineering Internship 2025.

---

## ✨ Key Takeaways

This implementation showcases:
- **Deep Learning**: Custom PyTorch modules, gradient computation
- **ML Engineering**: Complete training pipeline, evaluation, visualization
- **Code Quality**: Clean, documented, production-ready code
- **Problem Solving**: Elegant approach to network pruning
- **Analysis**: Comprehensive results interpretation

Perfect for demonstrating AI engineering fundamentals and practical skills! 🚀

---

**Developed for Tredence Analytics AI Engineering Internship 2025**
