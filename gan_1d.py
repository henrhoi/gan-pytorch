import time

import torch.distributions as dist

from utils.helper_functions import *


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class SimpleGAN(nn.Module):
    def __init__(self, latent_dim=5):
        super().__init__()
        self.latent_dim = latent_dim
        self.generator = nn.Sequential(MLP(latent_dim, 256, 1), nn.Tanh())
        self.discriminator = nn.Sequential(MLP(1, 256, 1), nn.Sigmoid())
        self.noise_prior = dist.uniform.Uniform(low=-1, high=1)  # dist.normal.Normal(loc=0, scale=1)

    def generate_x(self, no_samples):
        z = self.noise_prior.sample(sample_shape=(no_samples, self.latent_dim)).cuda()
        return self.generator(z)

    def calculate_loss(self, discriminator_fake, discriminator_true):
        return torch.mean(torch.log(discriminator_true + 1e-8) + torch.log(1 - discriminator_fake + 1e-8))

    def _model_iteration(self, x_samples):
        """
        Performing training iteration with k = 1
        """
        z = self.noise_prior.sample(sample_shape=(x_samples.size(0), self.latent_dim)).cuda()

        generated_x = self.generator(z)

        full_batch = torch.cat((x_samples, generated_x), dim=0)
        discriminator_true, discriminator_fake = torch.chunk(self.discriminator(full_batch), 2)
        return self.calculate_loss(discriminator_fake, discriminator_true)


def train_gan(train_data):
    """
    train_data: An (20000, 1) numpy array of floats in [-1, 1]

    Returns
    - a (# of training iterations,) numpy array of discriminator losses evaluated every minibatch
    - a numpy array of size (5000,) of samples drawn from your model at epoch #1
    - a numpy array of size (1000,) linearly spaced from [-1, 1]; hint: np.linspace
    - a numpy array of size (1000,), corresponding to the discriminator output (after sigmoid)
        at each location in the previous array at epoch #1

    - a numpy array of size (5000,) of samples drawn from your model at the end of training
    - a numpy array of size (1000,) linearly spaced from [-1, 1]; hint: np.linspace
    - a numpy array of size (1000,), corresponding to the discriminator output (after sigmoid)
        at each location in the previous array at the end of training
    """
    start_time = time.time()

    batch_size = 64  # This is m in algorithm provided
    dataset_params = {
        'batch_size': batch_size,
        'shuffle': True
    }

    train_loader = torch.utils.data.DataLoader(torch.from_numpy(train_data).float().cuda(), **dataset_params)

    # Model
    n_epochs = 25
    gan = SimpleGAN().cuda()

    generator_optimizer = torch.optim.Adam(gan.generator.parameters(), lr=1e-4, betas=(0, 0.9))
    discriminator_optimizer = torch.optim.Adam(gan.discriminator.parameters(), lr=1e-4, betas=(0, 0.9))

    # Training
    d_losses = []

    print("[INFO] Training")
    for epoch in range(n_epochs):
        if epoch == 1:
            # Generating output arrays (Part I)
            initial_samples = gan.generate_x(no_samples=5000).cpu().detach().numpy()
            linear_space = torch.from_numpy(np.linspace(-1, 1, 1000)).unsqueeze(1).float().cuda()
            initial_generator_output = gan.discriminator(linear_space).cpu().detach().numpy()

        epoch_start = time.time()
        for batch in train_loader:
            # Training discriminator
            discriminator_loss = -gan._model_iteration(batch)  # Discriminator maximizing
            discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            discriminator_optimizer.step()
            d_losses.append(discriminator_loss.cpu().item())

            # Training generator
            generator_loss = gan._model_iteration(batch)  # Generator minimizing
            generator_optimizer.zero_grad()
            generator_loss.backward()
            generator_optimizer.step()

        print(f"[{100*(epoch+1)/n_epochs:.2f}%] Epoch {epoch + 1} - Time elapsed: {time.time() - epoch_start:.2f}")

    d_losses = np.array(d_losses)

    # Generating output arrays (Part II)
    gan.eval()
    final_samples = gan.generate_x(no_samples=5000).cpu().detach().numpy()
    final_generator_output = gan.discriminator(linear_space).cpu().detach().numpy()
    linear_space = np.linspace(-1, 1, 1000)

    print(f"[DONE] Time elapsed: {time.time() - start_time:.2f} s")
    return d_losses, initial_samples, linear_space, initial_generator_output, final_samples, linear_space, final_generator_output


def train_and_show_results_1d_dataset():
    """
    Trains Simple GAN and displays samples and training plot for simple 1D dataset
    """
    show_results_1d(train_gan)
