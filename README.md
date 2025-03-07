# VAE
This project implements a Variational Autoencoder (VAE) model using TensorFlow and Keras to reconstruct and generate images for two datasets: Fashion MNIST (grayscale images) and CIFAR-10 (RGB images).
Project Overview

A Variational Autoencoder (VAE) is a type of generative model that learns a compressed latent representation of the input data and then generates new data samples from the learned distribution. This project includes:

    Encoder: A neural network that learns the mean (z_mean) and log-variance (z_log_var) of the latent space representation.
    Reparameterization Trick: The latent variable z is sampled from the learned distribution using the formula z = mean + sigma * epsilon.
    Decoder: A neural network that reconstructs the original image from the latent variable z.
    Loss Functions: A combination of reconstruction loss and KL divergence loss to train the model.
    The project includes models for both grayscale images (Fashion MNIST) and RGB images (CIFAR-10).

Requirements

    Python 3.x
    TensorFlow 2.x
    NumPy
    Matplotlib

To install the necessary packages, run:

pip install tensorflow numpy matplotlib

# Dataset

The project uses two datasets:

    Fashion MNIST: A dataset of grayscale images (28x28) of 10 clothing categories.
    CIFAR-10: A dataset of 32x32 RGB images across 10 classes.

Both datasets are loaded using tensorflow.keras.datasets.
# Model Architecture
Encoder

    Two convolutional layers followed by a dense layer.
    Outputs two vectors: z_mean (mean) and z_log_var (log-variance), which define the distribution of the latent space.

Sampling Layer

    Implements the reparameterization trick: z = z_mean + exp(0.5 * z_log_var) * epsilon.

Decoder

    A dense layer reshaped into a convolutional structure, followed by transpose convolution layers to reconstruct the image.

Loss Function

    Reconstruction Loss: Binary cross-entropy between original and reconstructed images.
    KL Divergence: Measures how much the learned latent distribution deviates from a normal distribution.

Training

    The model is trained using the Adam optimizer with a learning rate of 0.001.
    The loss is the sum of the weighted KL divergence loss and reconstruction loss.

# Results

The following plots are generated after training:

    Loss curves for total loss, reconstruction loss, and KL loss.
    Synthetic images generated from random latent vectors using the trained decoder.

Example Generated Images

Fashion MNIST example:

CIFAR-10 example:
Acknowledgments

This project is built using TensorFlow and Keras and utilizes publicly available datasets (Fashion MNIST and CIFAR-10).
