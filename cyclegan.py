import time

import torch.nn.functional as F
import torch.optim as optim

from utils.helper_functions import *


class ResidualBlock(nn.Module):
    """
    Residual block for CycleGAN
    """

    def __init__(self, in_features):
        super().__init__()

        self.net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3), nn.BatchNorm2d(in_features), nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3), nn.BatchNorm2d(in_features))

    def forward(self, x):
        return self.net(x) + x


class CycleGenerator(nn.Module):
    """
    Generator for CycleGAN
    """

    def __init__(self, input_channels, output_channels, n_residual_blocks=3):
        super().__init__()
        filters = 16

        # Convolution #1
        modules = [nn.ReflectionPad2d(3),
                   nn.Conv2d(input_channels, filters, kernel_size=7),
                   nn.BatchNorm2d(filters),
                   nn.ReLU()]

        # Downsampling conv layers
        for _ in range(2):
            modules += [nn.Conv2d(filters, filters * 2, kernel_size=3, stride=2, padding=1),
                        nn.BatchNorm2d(filters * 2),
                        nn.ReLU()]
            filters = filters * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            modules += [ResidualBlock(filters)]

        # Upsampling conv layers
        for _ in range(2):
            modules += [nn.ConvTranspose2d(filters, filters // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                        nn.BatchNorm2d(filters // 2),
                        nn.ReLU()]
            filters = filters // 2

        # Output layer
        modules += [nn.ReflectionPad2d(3),
                    nn.Conv2d(filters, output_channels, kernel_size=7),
                    nn.Tanh()]

        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


class CycleDiscriminator(nn.Module):
    """
    Discriminator for CycleGAN
    """

    def __init__(self, input_channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, padding=1), nn.InstanceNorm2d(512), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, padding=1))  # Lastly a classification layer

    def forward(self, x):
        x = self.net(x)
        # Average pooling
        avg_x = F.avg_pool2d(x, x.size()[2:])
        return avg_x.view(x.shape[0], -1)


def train_cycle_gan(mnist_data, cmnist_data):
    """
    mnist_data: An (60000, 1, 28, 28) numpy array of black and white images with values in [0, 1]
    cmnist_data: An (60000, 3, 28, 28) numpy array of colored images with values in [0, 1]

    Returns
    - a (20, 28, 28, 1) numpy array of real MNIST digits, in [0, 1]
    - a (20, 28, 28, 3) numpy array of translated Colored MNIST digits, in [0, 1]
    - a (20, 28, 28, 1) numpy array of reconstructed MNIST digits, in [0, 1]

    - a (20, 28, 28, 3) numpy array of real Colored MNIST digits, in [0, 1]
    - a (20, 28, 28, 1) numpy array of translated MNIST digits, in [0, 1]
    - a (20, 28, 28, 3) numpy array of reconstructed Colored MNIST digits, in [0, 1]
    """
    start_time = time.time()

    def normalize(data, reverse=False):
        if reverse:
            return (data + 1) / 2  # [-1, 1] to [ 0, 1]
        return (data - .5) / .5  # [ 0, 1] to [-1, 1]

    print("[INFO] Setting up")
    n_epochs = 5
    mnist_data = normalize(torch.from_numpy(mnist_data).float().cuda())
    cmnist_data = normalize(torch.from_numpy(cmnist_data).float().cuda())

    dataset_params = {'batch_size': 128, 'shuffle': True}
    mnist_loader = torch.utils.data.DataLoader(mnist_data, **dataset_params)
    cmnist_loader = torch.utils.data.DataLoader(cmnist_data, **dataset_params)
    assert (len(mnist_loader) == len(cmnist_loader))

    # Models - Assuming x is mnist and y is cmnist
    generator_x2y = CycleGenerator(input_channels=1, output_channels=3).cuda()
    generator_y2x = CycleGenerator(input_channels=3, output_channels=1).cuda()
    discriminator_x = CycleDiscriminator(input_channels=1).cuda()
    discriminator_y = CycleDiscriminator(input_channels=3).cuda()

    generator_optimizer = torch.optim.Adam(list(generator_x2y.parameters()) + list(generator_y2x.parameters()), lr=2e-4,
                                           betas=(0.5, 0.999))
    discriminator_x_optimizer = torch.optim.Adam(discriminator_x.parameters(), lr=2e-4, betas=(0.5, 0.999))
    discriminator_y_optimizer = torch.optim.Adam(discriminator_y.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # Schedulers
    num_batches = len(mnist_loader)
    max_steps = max(1, n_epochs * num_batches)

    def scheduler(epoch):
        return max(0, (max_steps - epoch * num_batches) / max_steps)

    generator_scheduler = optim.lr_scheduler.LambdaLR(generator_optimizer, scheduler)
    discriminator_x_scheduler = optim.lr_scheduler.LambdaLR(discriminator_x_optimizer, scheduler)
    discriminator_y_scheduler = optim.lr_scheduler.LambdaLR(discriminator_y_optimizer, scheduler)

    # Training
    print("[INFO] Training discriminator, generator and encoder")
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    zeros, ones = torch.tensor(0.).cuda(), torch.tensor(1.).cuda()
    cycle_lambda = 10

    for epoch in range(n_epochs):
        epoch_start = time.time()
        for x_batch, y_batch in zip(mnist_loader, cmnist_loader):
            # Training generators x2y and y2x
            x_fake = generator_y2x(y_batch)
            d_x_fake = discriminator_x(x_fake)
            generator_x_loss = mse_loss(d_x_fake, ones)

            y_fake = generator_x2y(x_batch)
            d_y_fake = discriminator_y(y_fake)
            generator_y_loss = mse_loss(d_y_fake, ones)

            x_recovered = generator_y2x(y_fake)
            y_recovered = generator_x2y(x_fake)
            cycle_loss = l1_loss(x_recovered, x_batch) + l1_loss(y_recovered, y_batch)

            generator_loss = generator_x_loss + generator_y_loss + cycle_lambda * cycle_loss
            generator_optimizer.zero_grad()
            generator_loss.backward()
            generator_optimizer.step()

            # Training discriminator for x
            x_fake = generator_y2x(y_batch)
            d_x_fake = discriminator_x(x_fake)
            d_x_true = discriminator_x(x_batch)
            discriminator_x_loss = mse_loss(d_x_true, ones) + mse_loss(d_x_fake, zeros)

            discriminator_x_optimizer.zero_grad()
            discriminator_x_loss.backward()
            discriminator_x_optimizer.step()

            # Training discriminator for y
            y_fake = generator_x2y(x_batch)
            d_y_fake = discriminator_y(y_fake)
            d_y_true = discriminator_y(y_batch)
            discriminator_y_loss = mse_loss(d_y_true, ones) + mse_loss(d_y_fake, zeros)

            discriminator_y_optimizer.zero_grad()
            discriminator_y_loss.backward()
            discriminator_y_optimizer.step()

        generator_scheduler.step()
        discriminator_x_scheduler.step()
        discriminator_y_scheduler.step()

        print(f"[{100*(epoch+1)/n_epochs:.2f}%] Epoch {epoch + 1} - Time elapsed: {time.time() - epoch_start:.2f}")

    # Generating samples
    generator_x2y.eval()
    generator_y2x.eval()
    discriminator_x.eval()
    discriminator_y.eval()
    with torch.no_grad():
        print("[INFO] Creating samples")
        mnist_samples = mnist_data[:20]
        mnist2cmnist = generator_x2y(mnist_samples)
        mnist_recovered = generator_y2x(generator_x2y(mnist_samples))

        cmnist_samples = cmnist_data[:20]
        cmnist2mnist = generator_y2x(cmnist_samples)
        cmnist_recovered = generator_x2y(generator_y2x(cmnist_samples))

        # Changing to desired output format
        mnist_samples = normalize(mnist_samples.permute(0, 2, 3, 1).cpu().numpy(), reverse=True)
        mnist2cmnist = normalize(mnist2cmnist.permute(0, 2, 3, 1).cpu().numpy(), reverse=True)
        mnist_recovered = normalize(mnist_recovered.permute(0, 2, 3, 1).cpu().numpy(), reverse=True)

        cmnist_samples = normalize(cmnist_samples.permute(0, 2, 3, 1).cpu().numpy(), reverse=True)
        cmnist2mnist = normalize(cmnist2mnist.permute(0, 2, 3, 1).cpu().numpy(), reverse=True)
        cmnist_recovered = normalize(cmnist_recovered.permute(0, 2, 3, 1).cpu().numpy(), reverse=True)

    print(f"[DONE] Time elapsed: {time.time() - start_time:.2f} s")
    return mnist_samples, mnist2cmnist, mnist_recovered, cmnist_samples, cmnist2mnist, cmnist_recovered


def train_and_show_results_mnist_and_colorized_mnist():
    """
    Trains CycleGAN and displays samples and training plot for MNIST and Colorized MNIST dataset
    """
    show_results_mnist_and_colorized_mnist(train_cycle_gan)
