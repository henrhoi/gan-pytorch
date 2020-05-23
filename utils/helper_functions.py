import math
import pickle
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytorch_util as ptu
import scipy.ndimage
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from PIL import Image as PILImage
from googlenet import *
from torchvision import transforms as transforms
from torchvision.utils import make_grid

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

softmax = None
model = None
device = torch.device("cuda:0")


def plot_gan_training(losses, title):
    plt.figure()
    n_itr = len(losses)
    xs = np.arange(n_itr)

    plt.plot(xs, losses, label='loss')
    plt.legend()
    plt.title(title)
    plt.xlabel('Training Iteration')
    plt.ylabel('Loss')


def q1_gan_plot(data, samples, xs, ys, title):
    plt.figure()
    plt.hist(samples, bins=50, density=True, alpha=0.7, label='fake')
    plt.hist(data, bins=50, density=True, alpha=0.7, label='real')

    plt.plot(xs, ys, label='discrim')
    plt.legend()
    plt.title(title)


# 1 Dimension data and result functions
def simple_1d_data(n=20000):
    assert n % 2 == 0
    gaussian1 = np.random.normal(loc=-1, scale=0.25, size=(n // 2,))
    gaussian2 = np.random.normal(loc=0.5, scale=0.5, size=(n // 2,))
    data = (np.concatenate([gaussian1, gaussian2]) + 1).reshape([-1, 1])
    scaled_data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
    return 2 * scaled_data - 1


def show_results_1d(fn):
    data = simple_1d_data()
    losses, samples1, xs1, ys1, samples_end, xs_end, ys_end = fn(data)

    # loss plot
    plot_gan_training(losses, 'Losses')

    # samples
    q1_gan_plot(data, samples1, xs1, ys1, 'Epoch 1')
    q1_gan_plot(data, samples_end, xs_end, ys_end, 'Final')


# CIFAR-10 helper functions
def calculate_is(samples):
    """
    Calculate inception score
    """
    assert (type(samples[0]) == np.ndarray)
    assert (len(samples[0].shape) == 3)

    model = GoogLeNet().to(ptu.device)
    model.load_state_dict(torch.load("./trained_googlenet.pt"))
    softmax = nn.Sequential(model, nn.Softmax(dim=1))

    bs = 100
    softmax.eval()
    with torch.no_grad():
        preds = []
        n_batches = int(math.ceil(float(len(samples)) / float(bs)))
        for i in range(n_batches):
            sys.stdout.write(".")
            sys.stdout.flush()
            inp = ptu.FloatTensor(samples[(i * bs):min((i + 1) * bs, len(samples))])
            pred = ptu.get_numpy(softmax(inp))
            preds.append(pred)
    preds = np.concatenate(preds, 0)
    kl = preds * (np.log(preds) - np.log(np.expand_dims(np.mean(preds, 0), 0)))
    kl = np.mean(np.sum(kl, 1))
    return np.exp(kl)


def load_cifar_data():
    train_data = torchvision.datasets.CIFAR10("./data", transform=torchvision.transforms.ToTensor(),
                                              download=True, train=True)
    return train_data


def show_results_cifar(fn):
    train_data = load_cifar_data()
    train_data = train_data.data.transpose((0, 3, 1, 2)) / 255.0
    train_losses, samples = fn(train_data)

    print("Inception score:", calculate_is(samples.transpose([0, 3, 1, 2])))
    plot_gan_training(train_losses, 'Losses')
    show_samples(samples[:100] * 255.0, title=f'CIFAR-10 generated samples')


# MNIST helper functions

def load_mnist_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_data = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_data = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    return train_data, test_data


def plot_classifiers_supervised(pretrained_losses, random_losses, title):
    plt.figure()
    xs = np.arange(len(pretrained_losses))
    plt.plot(xs, pretrained_losses, label='bigan')
    xs = np.arange(len(random_losses))
    plt.plot(xs, random_losses, label='random init')
    plt.legend()
    plt.title(title)


def show_results_mnist(fn):
    train_data, test_data = load_mnist_data()
    gan_losses, samples, reconstructions, pretrained_losses, random_losses = fn(train_data, test_data)

    plot_gan_training(gan_losses, 'Q3 Losses', 'results/q3_gan_losses.png')
    plot_classifiers_supervised(pretrained_losses, random_losses, 'Linear classification losses',
                                'results/q3_supervised_losses.png')
    show_samples(samples * 255.0, title='BiGAN generated samples')
    show_samples(reconstructions * 255.0, nrow=20, title=f'BiGAN reconstructions')
    print('BiGAN final linear classification loss:', pretrained_losses[-1])
    print('Random encoder linear classification loss:', random_losses[-1])


# Colored MNIST helper functions

def get_colored_mnist(data):
    # From https://www.wouterbulten.nl/blog/tech/getting-started-with-gans-2-colorful-mnist/
    # Read Lena image
    lena = PILImage.open('../images/lena.jpg')

    # Resize
    batch_resized = np.asarray([scipy.ndimage.zoom(image, (2.3, 2.3, 1), order=1) for image in data])

    # Extend to RGB
    batch_rgb = np.concatenate([batch_resized, batch_resized, batch_resized], axis=3)

    # Make binary
    batch_binary = (batch_rgb > 0.5)

    batch = np.zeros((data.shape[0], 28, 28, 3))

    for i in range(data.shape[0]):
        # Take a random crop of the Lena image (background)
        x_c = np.random.randint(0, lena.size[0] - 64)
        y_c = np.random.randint(0, lena.size[1] - 64)
        image = lena.crop((x_c, y_c, x_c + 64, y_c + 64))
        image = np.asarray(image) / 255.0

        # Invert the colors at the location of the number
        image[batch_binary[i]] = 1 - image[batch_binary[i]]

        batch[i] = cv2.resize(image, (0, 0), fx=28 / 64, fy=28 / 64, interpolation=cv2.INTER_AREA)
    return batch.transpose(0, 3, 1, 2)


def load_mnist_and_colorized_mnist_data():
    train, _ = load_mnist_data()
    mnist = np.array(train.data.reshape(-1, 28, 28, 1) / 255.0)
    colored_mnist = get_colored_mnist(mnist)
    return mnist.transpose(0, 3, 1, 2), colored_mnist


def show_results_mnist_and_colorized_mnist(fn):
    mnist, cmnist = load_mnist_and_colorized_mnist_data()

    m1, c1, m2, c2, m3, c3 = fn(mnist, cmnist)
    m1, m2, m3 = m1.repeat(3, axis=3), m2.repeat(3, axis=3), m3.repeat(3, axis=3)
    mnist_reconstructions = np.concatenate([m1, c1, m2], axis=0)
    colored_mnist_reconstructions = np.concatenate([c2, m3, c3], axis=0)

    show_samples(mnist_reconstructions * 255.0, nrow=20, title=f'Source domain: MNIST')
    show_samples(colored_mnist_reconstructions * 255.0, nrow=20, title=f'Source domain: Colored MNIST')
    pass


# General utils

def show_training_plot(train_losses, test_losses, title):
    plt.figure()
    n_epochs = len(test_losses) - 1
    x_train = np.linspace(0, n_epochs, len(train_losses))
    x_test = np.arange(n_epochs + 1)

    plt.plot(x_train, train_losses, label='train loss')
    plt.plot(x_test, test_losses, label='test loss')
    plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('NLL')
    plt.show()


def load_pickled_data(fname, include_labels=False):
    with open(fname, 'rb') as f:
        data = pickle.load(f)

    train_data, test_data = data['train'], data['test']
    if 'mnist.pkl' in fname or 'shapes.pkl' in fname:
        # Binarize MNIST and shapes dataset
        train_data = (train_data > 127.5).astype('uint8')
        test_data = (test_data > 127.5).astype('uint8')
    if 'celeb.pkl' in fname:
        train_data = train_data[:, :, :, [2, 1, 0]]
        test_data = test_data[:, :, :, [2, 1, 0]]
    if include_labels:
        return train_data, test_data, data['train_labels'], data['test_labels']
    return train_data, test_data


def show_samples(samples, nrow=10, title='Samples'):
    samples = (torch.FloatTensor(samples) / 255).permute(0, 3, 1, 2)
    grid_img = make_grid(samples, nrow=nrow)
    plt.figure()
    plt.title(title)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis('off')
    plt.show()
