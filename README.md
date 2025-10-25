# MNIST Classification with Custom ResNet
This project demonstrates how to load, preprocess, and train a convolutional neural network on the MNIST dataset.
The dataset is loaded manually from the original binary `.idx` files without using built-in Keras datasets.
The model is based on a custom lightweight ResNet architecture and uses several modern training techniques to achieve high accuracy.

---

## Features

* Custom MNIST data loader (reads raw `.idx` files)
* Data preprocessing and normalization
* Data augmentation using `ImageDataGenerator`
* MixUp data regularization
* Custom ResNet-like CNN model
* One-Cycle learning rate scheduler
* AdamW optimizer with weight decay
* Label smoothing for regularization
* Early stopping to prevent overfitting
* Visualization of training and validation metrics

---

## Model Architecture

The model is a small ResNet-style CNN designed for MNIST:

1. Conv2D(32) + BatchNorm + ReLU
2. Residual block (32 filters) + MaxPooling
3. Residual block (64 filters) + MaxPooling
4. Residual block (128 filters) + GlobalAveragePooling
5. Dense(128, ReLU) + Dropout(0.5)
6. Dense(10, Softmax)

Each residual block uses two convolutional layers with skip connections.

---

## Training Details

| Setting              | Value                                               |
| -------------------- | --------------------------------------------------- |
| Optimizer            | AdamW (weight decay = 1e-4)                         |
| Learning rate policy | One-Cycle (1e-5 → 1e-3 → 1e-5)                      |
| Loss                 | Categorical crossentropy with label smoothing (0.1) |
| Batch size           | 128                                                 |
| Epochs               | 150 (with early stopping)                           |
| Data augmentation    | rotation ±15°, shift ±15%, zoom/shear 0.15          |

---

## Results

The final test accuracy reaches about **99.4–99.8%**, depending on the random seed and training conditions.
The model trains quickly and generalizes well due to the combination of MixUp, label smoothing, and One-Cycle learning rate scheduling.

---

## Visualization

At the end of training, three plots are displayed:

* Training vs validation accuracy
* Training vs validation loss
* Learning rate schedule

---

## Project Structure

```
MNIST-ResNet/
│
├── trainmnist.ipynb   # Jupyter Notebook with the training pipeline
├── train-images.idx3-ubyte    # MNIST training images
├── train-labels.idx1-ubyte    # MNIST training labels
├── t10k-images.idx3-ubyte     # MNIST test images
├── t10k-labels.idx1-ubyte     # MNIST test labels
└── README.md                  # Project description
```

---

## Requirements

```
pip install numpy matplotlib tensorflow
```

---

## How to Run

1. Download the original MNIST `.idx` files.
2. Place them in the same folder as the notebook.
3. Open `trainmnist.ipynb` in Colab or VSCode.
4. Run all cells to start training.

---

## Notes
# tg: @armaturablya
student RUDN University
This project was created for learning purposes.
It shows how to build and train a deep learning model from scratch using TensorFlow, including manual dataset loading, data augmentation, and advanced optimization techniques.
