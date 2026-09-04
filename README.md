# TensorFlow Deep Learning Basics

A beginner-friendly deep learning project demonstrating synthetic data generation using NumPy and neural network model training with TensorFlow and Keras.

## Project Overview

This project provides a practical introduction to deep learning fundamentals, showcasing:
- Synthetic dataset generation with controlled parameters
- Simple linear regression using neural networks
- Model training and weight extraction
- TensorFlow/Keras workflow basics

## Objectives

The primary objective is to demonstrate how to:
1. Generate synthetic training data programmatically
2. Build and compile a neural network model
3. Train the model using Stochastic Gradient Descent (SGD) optimization
4. Extract and interpret learned model weights and biases

## Dataset

### Data Generation

The dataset is synthetically generated using NumPy with the following specifications:

- **Sample Size**: 1,000 observations
- **Input Features**: 2 continuous variables
  - `xs`: Uniformly distributed values in range [-10, 10]
  - `zs`: Uniformly distributed values in range [-10, 10]
- **Target Generation**: Linear relationship with noise
  - Formula: `y = 2*xs - 3*zs + 5 + noise`
  - Noise: Uniformly distributed in range [-1, 1]

### Data Format

The generated dataset is stored in NumPy's compressed archive format (`TF_intro.npz`), containing:
- `inputs`: Array of shape (1000, 2) containing feature pairs
- `targets`: Array of shape (1000, 1) containing target values

**How to Generate the Data:**

```python
import numpy as np

observations = 1000
xs = np.random.uniform(low=-10, high=10, size=(observations, 1))
zs = np.random.uniform(-10, 10, (observations, 1))
generated_inputs = np.column_stack((xs, zs))
noise = np.random.uniform(-1, 1, (observations, 1))
generated_targets = 2*xs - 3*zs + 5 + noise

np.savez('TF_intro', inputs=generated_inputs, targets=generated_targets)
```

## Preprocessing

No explicit preprocessing steps are required. The synthetic data is generated with features already in a suitable numerical range [-10, 10]. The model operates on raw continuous features without scaling or normalization.

## Deep Learning Model

### Architecture

A simple feedforward neural network implemented using Keras Sequential API:

- **Input Layer**: 2 features (xs, zs)
- **Output Layer**: 1 unit (Dense layer with linear activation)
- **Total Parameters**: 3 (2 weights + 1 bias)

### Training Configuration

- **Optimizer**: Stochastic Gradient Descent (SGD)
- **Loss Function**: Mean Squared Error (MSE)
- **Batch Size**: 32 (default)
- **Epochs**: 100
- **Verbose**: Epoch-by-epoch loss reporting enabled

### Model Code

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(training_data['inputs'], training_data['targets'], epochs=100, verbose=1)
```

## Evaluation Results

### Training Progress

The model demonstrates effective learning throughout 100 epochs:

| Epoch | Training Loss |
|-------|---------------|
| 1     | 38.5694       |
| 2     | 4.4910        |
| 3     | 1.5215        |
| 10    | 0.3413        |
| 50    | 0.3411        |
| 100   | 0.3492        |

### Learned Parameters

After 100 epochs of training, the model converged to the following weights and bias:

**Weight Matrix** (2×1):
- Weight for xs: **1.938425**
- Weight for zs: **-3.0642006**

**Bias**: **5.0281687**

### Performance Analysis

The model successfully learned weights very close to the true underlying linear relationship:
- True coefficients: [2, -3] with bias 5
- Learned coefficients: [1.938, -3.064] with bias 5.028
- The small discrepancies are due to the noise added during synthetic data generation

The loss converges from 38.57 to approximately 0.34 over 100 epochs, indicating successful model learning of the target function.

## Technologies Used

- **Python 3.x**: Programming language
- **NumPy**: Synthetic data generation and array operations
- **TensorFlow 2.x**: Deep learning framework
- **Keras**: High-level neural network API (integrated with TensorFlow)
- **Matplotlib**: Data visualization (imported but not utilized in core training loop)

## Project Structure

```
Deep Learning-Tensorflow.ipynb    Main Jupyter notebook with complete workflow
README.md                         This documentation file
.gitignore                        Git ignore patterns
```

## Running the Project

1. **Prerequisites**: Install required packages:
   ```bash
   pip install numpy tensorflow matplotlib
   ```

2. **Execute the Notebook**: Open `Deep Learning-Tensorflow.ipynb` in Jupyter Notebook or JupyterLab

3. **Workflow**:
   - Cell 1: Import libraries
   - Cell 2: Generate synthetic dataset
   - Cell 3: Load the generated data
   - Cell 4: Build, compile, and train the model
   - Cell 5-9: Extract and display learned weights and bias

## Learning Outcomes

By working through this project, you will understand:
- How to generate and structure synthetic datasets
- The fundamentals of neural network architecture
- How to use TensorFlow/Keras for model training
- How optimization algorithms (SGD) minimize loss
- How to extract and interpret learned model parameters

## Notes

- This project uses synthetic data with a known linear relationship, making it ideal for educational purposes
- The small prediction error remaining after training is attributed to the random noise included in the dataset
- The simple architecture (single Dense layer) is sufficient for this linear regression task

## License

This project is provided as-is for educational purposes.
