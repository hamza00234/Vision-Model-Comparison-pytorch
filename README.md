Vision Model Comparison (PyTorch)
Overview

A hands-on notebook comparing simple neural network architectures for multiclass image classification using PyTorch.
Implements and compares:
A baseline linear model (fully connected)
A simple MLP with ReLU nonlinearity
A small CNN (TinyVGG-style)
Uses the FashionMNIST dataset from torchvision to demonstrate dataset loading, batching, training, evaluation, and simple model comparisons.
Notebook

computerVision (3).ipynb — the main Jupyter/Colab notebook containing data preparation, model definitions, training loops, evaluation functions, timing utilities, and comparisons.
What this project covers

Vision datasets and transforms (torchvision, ToTensor)
DataLoader and batching
Building models in PyTorch (nn.Module, Sequential)
Baseline linear model vs. MLP vs. CNN
Training loop, evaluation loop, accuracy via torchmetrics
Visualizing sample images and inspecting layer shapes
Comparing model performance and basic metrics reporting
Requirements

Python 3.8+
PyTorch (cpu / cuda) and torchvision
matplotlib
torchmetrics
tqdm
pandas
Quick install (example, CPU)

It's recommended to install PyTorch following instructions on https://pytorch.org for your environment. Example (CPU-only pip): pip install torch torchvision matplotlib torchmetrics tqdm pandas
Usage

Clone the repo or download the notebook.
Open the notebook in Jupyter Notebook / JupyterLab or upload to Google Colab.
Run the cells top-to-bottom. The FashionMNIST dataset is downloaded automatically by torchvision.
Experiment by changing:
Model hyperparameters (hidden units, number of filters)
Learning rate and optimizer
Number of epochs and batch size
Add augmentation transforms in torchvision.transforms
Main files and structure

computerVision (3).ipynb — main notebook (contains:
Data loading and transforms
Baseline model (fashion_model0)
MLP with ReLU (fashion_model1)
TinyVGG-style CNN (FasionCnn)
train/test/eval helper functions
model comparisons and metrics)
data/ (created automatically by torchvision when downloading FashionMNIST)
Key implementation notes

Metrics: torchmetrics.MulticlassAccuracy is used to compute accuracy for the 10 FashionMNIST classes.
Training loops: notebook contains both inline training loop and modular train_step/test_step/eval_model helpers.
CNN in the notebook demonstrates how convolution, ReLU activations and MaxPool layers reduce spatial dimensions; classifier uses a Flatten layer and a final Linear layer with in_features calculated from conv output shape.
Reproducing experiments

Set torch.manual_seed(42) to reproduce deterministic runs in the notebook (as shown).
Use the provided DataLoader, BATCH_SIZE=32, and the example training loops/optimizers in the notebook.
The notebook trains each model for a small number of epochs (default 3) for quick demonstration. Increase epochs for better final performance.
Example hyperparameters used in notebook

Batch size: 32
Optimizer: SGD with lr=0.1
Loss: CrossEntropyLoss
Metrics: MulticlassAccuracy (torchmetrics)
Results

The notebook computes loss and accuracy for each model and builds a pandas DataFrame comparing model metrics.
Use the notebook’s evaluation and plotting cells to visualize results, confusion matrices, and predictions.
Tips & Next steps

Add data augmentations (RandomHorizontalFlip, RandomRotation) to improve generalization.
Try pretrained models from torchvision.models (transfer learning) for better accuracy on small datasets.
Replace SGD with Adam for faster convergence in many cases.
Add learning-rate scheduling and weight decay (L2 regularization).
Save and load trained models with torch.save / torch.load.
Contributing

Suggestions, improvements, and pull requests are welcome. If you want the README committed directly to the repo, tell me and I can prepare a commit message and add it for you.
License

Suggested: MIT License (replace with your preferred license)
Acknowledgements

FashionMNIST dataset via torchvision.datasets
PyTorch and torchvision documentation and examples
