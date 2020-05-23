import time

import torch.optim as optim

from utils.helper_functions import *


class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        t_1 = output.reshape(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        spl = t_1.split(self.block_size, 3)
        stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).reshape(batch_size, s_height, s_width,
                                                                                      s_depth)
        output = output.permute(0, 3, 1, 2)
        return output.contiguous()


class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output.contiguous()


class UpsampleConv2d(nn.Module):
    """
    Spatial Upsampling with Nearest Neighbors
    """

    def __init__(self, in_dim, out_dim, kernel_size=(3, 3), stride=1, padding=1):
        super().__init__()
        self.depthToSpace = DepthToSpace(block_size=2)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = torch.cat([x, x, x, x], dim=1)
        out = self.depthToSpace(x)
        return self.conv(out)


class DownsampleConv2d(nn.Module):
    """
    Spatial Downsampling with Spatial Mean Pooling
    """

    def __init__(self, in_dim, out_dim, kernel_size=(3, 3), stride=1, padding=1):
        super().__init__()
        self.spaceToDepth = SpaceToDepth(2)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        out = self.spaceToDepth(x)
        out = torch.sum(torch.stack(out.chunk(4, dim=1)), dim=0) / 4.0
        return self.conv(out)


class ResnetBlockUp(nn.Module):
    def __init__(self, in_dim, kernel_size=(3, 3), n_filters=256):
        super().__init__()
        self.residual_block = nn.Sequential(
            nn.BatchNorm2d(in_dim), nn.ReLU(),
            nn.Conv2d(in_dim, n_filters, kernel_size, padding=1), nn.BatchNorm2d(n_filters), nn.ReLU(),
            UpsampleConv2d(n_filters, n_filters, kernel_size, padding=1))

        self.shortcut_block = UpsampleConv2d(in_dim, n_filters, kernel_size=(1, 1), padding=0)

    def forward(self, x):
        residual = self.residual_block(x)
        shortcut = self.shortcut_block(x)
        return residual + shortcut


class ResBlockDown(nn.Module):
    def __init__(self, in_dim, kernel_size=(3, 3), n_filters=256):
        super().__init__()
        self.residual_block = nn.Sequential(
            nn.Conv2d(in_dim, n_filters, kernel_size, padding=1), nn.ReLU(),
            DownsampleConv2d(n_filters, n_filters, kernel_size, padding=1))

        self.shortcut_block = DownsampleConv2d(in_dim, n_filters, kernel_size=(1, 1), padding=0)

    def forward(self, x):
        residual = self.residual_block(x)
        shortcut = self.shortcut_block(x)
        return residual + shortcut


class Reshape(nn.Module):
    """
    Reshaping module for Generator and Discriminator
    """

    def __init__(self, flatten_input=False):
        super().__init__()
        self.flatten_input = flatten_input

    def forward(self, x):
        if self.flatten_input:
            return x.reshape(x.shape[0], -1)  # (N, C * H * W)
        else:
            return x.reshape(x.shape[0], 256, 4, 4)  # (N, C, H, W)


class CifarGenerator(nn.Module):
    """
    Generator for CIFAR10
    """

    def __init__(self, n_filters=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(128, 4 * 4 * 256),
            Reshape(),
            ResnetBlockUp(in_dim=256, n_filters=n_filters),
            ResnetBlockUp(in_dim=n_filters, n_filters=n_filters),
            ResnetBlockUp(in_dim=n_filters, n_filters=n_filters),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            nn.Conv2d(n_filters, 3, kernel_size=(3, 3), padding=1),
            nn.Tanh())

    def forward(self, n_samples):
        z = torch.randn((n_samples, 128)).cuda()
        return self.net(z)


class CifarDiscriminator(nn.Module):
    """
    Discriminator for CIFAR10
    """

    def __init__(self, n_filters=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, n_filters, kernel_size=(3, 3), padding=1), nn.ReLU(),
            ResBlockDown(in_dim=n_filters, n_filters=n_filters),
            ResBlockDown(in_dim=n_filters, n_filters=n_filters),
            ResBlockDown(in_dim=n_filters, n_filters=n_filters))
        # Reshape(flatten_input=True), nn.Linear(4*4*256, 1))
        self.linear = nn.Linear(128, 1)

    def forward(self, x):
        tmp = torch.mean(self.net(x), dim=(2, 3))  # (B, 128)
        return self.linear(tmp)


def train_wgan_gp(train_data):
    """
    train_data: An (n_train, 3, 32, 32) numpy array of CIFAR-10 images with values in [0, 1]

    Returns
    - a (# of training iterations,) numpy array of WGAN critic train losses evaluated every minibatch
    - a (1000, 32, 32, 3) numpy array of samples from your model in [0, 1].
        The first 100 will be displayed, and the rest will be used to calculate the Inception score.
    """
    start_time = time.time()

    def normalize(data, reverse=False):
        if reverse:
            return (data + 1) / 2  # we go from [-1, 1] to [0, 1]
        return (data - .5) / .5

    train_data = normalize(torch.from_numpy(train_data).float().cuda())

    n_epochs = 128  # 128 epochs will be the required number of gradient updates: (gradient steps / batch size) = (25000 / 196) ≈ 128
    dataset_params = {
        'batch_size': 256,
    }

    train_loader = torch.utils.data.DataLoader(train_data, **dataset_params)
    num_batches = len(train_loader)
    max_steps = n_epochs * num_batches

    # Model
    lambda_val = 10
    ncritic = 5
    n_filters = 128
    generator = CifarGenerator(n_filters).cuda()  # Output is in [-1, 1]
    discriminator = CifarDiscriminator(n_filters).cuda()  # Input is in [-1, 1]

    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0, 0.9))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0, 0.9))

    def scheduler(epoch):
        return max(0, (max_steps - epoch * num_batches) / max_steps)

    generator_scheduler = optim.lr_scheduler.LambdaLR(generator_optimizer, scheduler)
    discriminator_scheduler = optim.lr_scheduler.LambdaLR(discriminator_optimizer, scheduler)

    # Training
    save_every = 10
    print("[INFO] Training")
    d_losses = []
    for epoch in range(n_epochs):
        epoch_start = time.time()
        for i, batch in enumerate(train_loader):
            batch_size = batch.shape[0]

            d_true = discriminator(batch)
            x_fake = generator(n_samples=batch_size)
            d_fake = discriminator(x_fake)

            eps = torch.rand((batch_size, 1, 1, 1)).cuda()
            x_hat = eps * batch + (1. - eps) * x_fake
            x_hat = torch.autograd.Variable(x_hat, requires_grad=True)

            d_xhat = discriminator(x_hat)
            grad = torch.autograd.grad(outputs=d_xhat, inputs=x_hat,
                                       grad_outputs=torch.ones_like(d_xhat),
                                       create_graph=True,
                                       retain_graph=True)[0]

            grad = grad.view(grad.shape[0], -1)
            grad_norm = ((grad.norm(2, dim=1) - 1) ** 2) * lambda_val
            grad_norm = grad_norm.mean()

            discriminator_optimizer.zero_grad()
            discriminator_loss = d_fake.mean() - d_true.mean() + grad_norm
            discriminator_loss.backward()
            discriminator_optimizer.step()

            if i % ncritic == 0:
                x_fake = generator(n_samples=batch_size)
                d_fake = discriminator(x_fake)
                generator_loss = - torch.mean(d_fake)

                generator_optimizer.zero_grad()
                generator_loss.backward()
                generator_optimizer.step()

            d_losses.append(discriminator_loss.item())

        generator_scheduler.step()
        discriminator_scheduler.step()

        print(
            f"[{100*(epoch+1)/n_epochs:.2f}%] Epoch {epoch + 1} - Loss: {d_losses[-1]:.3f} - Time elapsed: {time.time() - epoch_start:.2f}")

    d_losses = np.array(d_losses)

    # Generating samples
    generator.eval()
    discriminator.eval()
    with torch.no_grad():
        x_fake = generator(n_samples=1000)  # Shape is (N, C, H, W) in [-1, 1]
        samples = normalize(x_fake, reverse=True)  # Pixel-values are in [0, 1]
        samples = samples.permute(0, 2, 3, 1).cpu().detach().numpy()  # Shape is (N, H, W, C)

    print(f"[DONE] Time elapsed: {time.time() - start_time:.2f} s")
    return d_losses, samples


def train_and_show_results_1d_dataset():
    """
    Trains WGAN-GP and displays samples and training plot for CIFAR-10 dataset
    """
    show_results_cifar(train_wgan_gp)

