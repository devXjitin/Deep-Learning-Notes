# DEEP LEARNING SYLLABUS (Beginner → Advanced)
---

## Introduction to Deep Learning

* What is Artificial Intelligence, Machine Learning, and Deep Learning
* Why deep learning is needed for complex problems
* Difference between traditional ML and deep learning
* Real-world applications of deep learning
* Advantages and limitations of deep learning
* Overview of deep learning workflow

---

## Neural Networks

* Biological neuron vs artificial neuron
* Structure of an artificial neuron (inputs, weights, bias)
* Weighted sum and output calculation
* Single-layer neural network
* Multi-layer neural network intuition
* Role of layers in learning representations

---

## Deep Learning vs Statistical Machine Learning

* Feature engineering vs feature learning
* Data dependency comparison
* Model interpretability differences
* Performance on structured vs unstructured data
* Scalability with large datasets

---

## Neural Network Architectures

* Shallow vs deep networks
* Fully connected (dense) networks
* Input, hidden, and output layers
* How depth improves representation power
* Common architectural design patterns

---

## Applications of Deep Learning

* Image classification and object detection
* Speech recognition
* Natural language processing
* Recommendation systems
* Time-series prediction
* Autonomous systems

---

## PyTorch vs TensorFlow

* Overview of deep learning frameworks
* Static vs dynamic computation graphs
* Ease of debugging and experimentation
* Research vs production use cases
* Community and ecosystem comparison

---

## PyTorch Installation

* Python environment setup
* Installing PyTorch with CPU support
* Installing PyTorch with GPU (CUDA) support
* Verifying installation
* Common installation issues

---

## PyTorch Tensor Basics

* What is a tensor
* Tensor dimensions and shapes
* Creating tensors
* Tensor arithmetic operations
* Broadcasting rules
* Converting NumPy arrays to tensors
* Moving tensors between CPU and GPU

---

## Activation Functions

* Why activation functions are required
* Linear vs non-linear transformations
* Sigmoid activation function
* Tanh activation function
* ReLU activation function
* Softmax activation function
* Choosing activation functions for output layers

---

## GPU and TPU

* Why hardware acceleration is needed
* CPU vs GPU computation
* Basics of CUDA
* When to use GPU vs CPU
* Handling device placement in PyTorch

---

## Neuron, Perceptron, and MLP

* Perceptron model
* Decision boundaries
* XOR problem
* Multi-layer perceptron (MLP)
* Hidden layer role in learning non-linearity

---

## Autograd in PyTorch

* Automatic differentiation concept
* Computational graph
* requires_grad usage
* Backward propagation using autograd
* Gradient accumulation
* Detaching tensors from graph

---

## Training Through Backpropagation

* Forward pass computation
* Loss calculation
* Gradient flow from output to input
* Chain rule intuition
* Weight update mechanism

---

## Gradient Descent

* Optimization problem formulation
* Loss surface intuition
* Learning rate selection
* Convergence behavior
* Local minima and saddle points

---

## Batch GD vs Mini-Batch GD vs SGD

* Batch gradient descent workflow
* Mini-batch gradient descent advantages
* Stochastic gradient descent behavior
* Trade-offs between speed and stability
* Choosing batch size

---

## Handwritten Digits Classification

* Understanding image data as numbers
* Dataset loading and preprocessing
* Normalization of pixel values
* Building an MLP classifier
* Training loop implementation
* Evaluating model performance
* Error analysis

---

## Model Optimization

* Why training can be slow or unstable
* Learning rate scheduling
* Gradient clipping
* Weight initialization impact
* Improving convergence speed

---

## Gradient Descent with Momentum

* Momentum intuition
* Velocity-based updates
* Faster convergence behavior
* Avoiding oscillations during training

---

## Adam Optimizer

* Adaptive learning rates
* Bias correction
* Adam vs SGD comparison
* Practical usage in PyTorch

---

## Regularization

* Overfitting and underfitting
* Generalization concept
* L1 regularization
* L2 regularization (weight decay)
* Regularization impact on model weights

---

## Dropout Regularization

* Neuron co-adaptation problem
* Random neuron dropping
* Dropout rate selection
* Training vs inference behavior

---

## Batch Normalization

* Internal covariate shift
* Normalizing layer inputs
* Faster and stable training
* Placement of batch normalization layers
* Training vs inference differences

---

## Hyperparameter Tuning

* What are hyperparameters
* Learning rate tuning
* Batch size tuning
* Network depth and width tuning
* Manual tuning strategies
* Automated tuning with Optuna

---

## Convolutional Neural Networks (CNN)

* Why CNNs are used for images
* Convolution operation intuition
* Filters and kernels
* Feature maps
* Pooling layers
* CNN architecture flow

---

## CIFAR-10 Image Classification Using CNN

* Dataset structure
* Data preprocessing
* CNN model design
* Training and validation
* Accuracy improvement techniques

---

## Data Augmentation

* Why data augmentation is needed
* Common image augmentation techniques
* Preventing overfitting
* Augmentation pipelines in PyTorch

---

## Recurrent Neural Networks (RNN)

* Sequential data understanding
* Time-step based processing
* Hidden state concept
* RNN training challenges

---

## Vanishing Gradient Problem

* Gradient decay in deep networks
* Why RNNs suffer more
* Effect on long-term dependencies

---

## Transfer Learning

* Concept of pre-trained models
* Feature extraction
* Fine-tuning strategies
* Domain similarity importance

---

## Pre-trained Models

* ResNet architecture overview
* EfficientNet scaling concept
* MobileNet for lightweight models
* Loading and customizing models

---

## LSTM

* Long-term dependency problem
* Memory cell concept
* Forget, input, and output gates
* Sequence modeling with LSTM

---

## Transformer Architecture

* Motivation behind transformers
* Encoder-decoder structure
* Parallel processing advantage
* Comparison with RNN-based models

---

## Word Embeddings

* One-hot encoding limitations
* Dense vector representations
* Semantic similarity
* Using embeddings in models

---

## Attention Mechanism

* Query, Key, and Value concept
* Self-attention
* Importance of attention in transformers

---

## Hugging Face: BERT Basics

* Hugging Face ecosystem overview
* Tokenizers and encoding
* Pre-trained BERT models
* Fine-tuning BERT for tasks
* Inference using pipelines

---

## Model Training with CNN & Transfer Learning

* Training loop structure
* Loss monitoring
* Validation strategies
* Avoiding overfitting
* Performance evaluation

---