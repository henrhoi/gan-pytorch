import time

import torch.distributions as dist
import torch.optim as optim

from utils.helper_functions import *


class BiGANGenerator(nn.Module):
    """
    Generator for BiGAN
    """

    def __init__(self, latent_dim=50, hidden_dim=1024):
        super().__init__()
        self.latent_dim = latent_dim
        self.z_prior = dist.normal.Normal(0., 1.)
        self.net = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=hidden_dim), nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim), nn.BatchNorm1d(hidden_dim, affine=False),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=784), nn.Tanh())

    def forward(self, z=None, num_samples=100):
        """
        Outputting G(z), z ~ N(0, 1)
        """
        if z is None:
            z = self.z_prior.sample(sample_shape=(num_samples, self.latent_dim)).cuda()
        return self.net(z), z


class BiGANDiscriminator(nn.Module):
    """
    Discriminator for BiGAN
    """

    def __init__(self, in_features=28 * 28 + 50, hidden_dim=1024):
        """
        Initialization:
            In features defaults to flattened and concatenated result of image x (N, 28, 28) and latent variable z (N, 50).
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_dim), nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim), nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=hidden_dim, out_features=1), nn.Sigmoid())

    def forward(self, x):
        """
        Outputting D(x), x in R^50
        """
        return self.net(x)  # this is D(x) or D(G(z))


class Encoder(nn.Module):
    """
    Encoder for BiGAN
    """

    def __init__(self, in_features=28 * 28, latent_dim=50, hidden_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_dim), nn.LeakyReLU(negative_slope=.2),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim), nn.BatchNorm1d(hidden_dim, affine=False),
            nn.LeakyReLU(negative_slope=.2),
            nn.Linear(in_features=hidden_dim, out_features=latent_dim))

    def forward(self, x):
        """
        Outputting E(x) ≈ z
        """
        return self.net(x)


class LinearClassifier(nn.Module):
    def __init__(self, latent_dim=50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=10),
            nn.Softmax())

        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        Outputting y ≈ L(E(x)), which is label y in {0, ..., 9} of image x
        """
        return self.net(x)

    def cross_entropy_loss(self, z, labels):
        logit_probs = self(z)
        return self.loss_fct(logit_probs, labels)


def train_bigan(train_data, test_data):
    """
    train_data: A PyTorch dataset that contains (n_train, 28, 28) MNIST digits, normalized to [-1, 1]
                Documentation can be found at torchvision.datasets.MNIST, and it may be easiest to directly create a DataLoader from this variable
    test_data: A PyTorch dataset that contains (n_test, 28, 28) MNIST digits, normalized to [-1, 1]
                Documentation can be found at torchvision.datasets.MNIST

    Returns
    - a (# of training iterations,) numpy array of BiGAN minimax losses evaluated every minibatch
    - a (100, 28, 28, 1) numpy array of BiGAN samples that lie in [0, 1]
    - a (40, 28, 28, 1) numpy array of 20 real image / reconstruction pairs
    - a (# of training epochs,) numpy array of supervised cross-entropy losses on the BiGAN encoder evaluated every epoch
    - a (# of training epochs,) numpy array of supervised cross-entropy losses on a random encoder evaluated every epoch
    """
    start_time = time.time()

    def reverse_normalize(data):
        return (data + 1) / 2  # we go from [-1, 1] to [0, 1]

    def scale(data):
        low, high = -1, 1
        return (high - low) * ((data - data.min()) / (data.max() - data.min())) + low

    print("[INFO] Setting up")
    train_images = scale(train_data.data.float())
    test_images = scale(test_data.data.float())
    n_epochs = 256
    lc_epochs = 30
    dataset_params = {'batch_size': 128}

    test_images, test_labels = torch.flatten(test_images, start_dim=1).cuda(), test_data.targets.cuda()
    train_loader = torch.utils.data.DataLoader(torch.flatten(train_images, start_dim=1).cuda(), **dataset_params)
    train_loader_with_labels = torch.utils.data.DataLoader(train_data, **dataset_params)
    test_loader = torch.utils.data.DataLoader(test_data, **dataset_params)

    num_batches = len(train_loader)
    max_steps = n_epochs * num_batches

    # Model
    generator = BiGANGenerator().cuda()  # Output is in [-1, 1]
    discriminator = BiGANDiscriminator().cuda()  # Input is in [-1, 1]
    linear_classifier = LinearClassifier().cuda()
    rnd_linear_classifier = LinearClassifier().cuda()
    encoder = Encoder().cuda()
    rnd_encoder = Encoder().cuda()
    rnd_encoder.eval()

    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    generator_encoder_optimizer = torch.optim.Adam(list(generator.parameters()) + list(encoder.parameters()), lr=2e-4,
                                                   betas=(0.5, 0.999))
    classifier_optimizer = torch.optim.Adam(linear_classifier.parameters(), lr=2e-4)
    rnd_classifier_optimizer = torch.optim.Adam(rnd_linear_classifier.parameters(), lr=2e-4)

    def scheduler(epoch):
        return max(0, (max_steps - epoch * num_batches) / max_steps)

    discriminator_scheduler = optim.lr_scheduler.LambdaLR(discriminator_optimizer, scheduler)
    generator_encoder_scheduler = optim.lr_scheduler.LambdaLR(generator_encoder_optimizer, scheduler)

    def minimax_loss(x_true, z_true):
        N = x_true.shape[0]
        x_fake, z_fake = generator(num_samples=N)
        d_fake = discriminator(torch.cat((x_fake, z_fake), dim=1))
        d_true = discriminator(torch.cat((x_true, z_true), dim=1))
        objective = torch.mean(torch.log(d_true + 1e-8) + torch.log(1 - d_fake + 1e-8))
        return objective

    # Training
    print("[INFO] Training discriminator, generator and encoder")
    d_losses = []
    classifier_losses = []
    rnd_classifier_losses = []
    for epoch in range(n_epochs):
        epoch_start = time.time()
        for x_true in train_loader:
            # Training discriminator
            z_true = encoder(x_true)
            discriminator_loss = -minimax_loss(x_true, z_true)

            discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            discriminator_optimizer.step()
            d_losses.append(discriminator_loss.item())

            # Training generator and encoder
            z_true = encoder(x_true)
            generator_encoder_loss = minimax_loss(x_true, z_true)

            generator_encoder_optimizer.zero_grad()
            generator_encoder_loss.backward()
            generator_encoder_optimizer.step()

        discriminator_scheduler.step()
        generator_encoder_scheduler.step()

        print(
            f"[{100*(epoch+1)/n_epochs:.2f}%] Epoch {epoch + 1} - Loss: {d_losses[-1]:.3f} - Time elapsed: {time.time() - epoch_start:.2f}")

    print("[INFO] Training linear classifiers")
    # Training linear classifier
    for epoch in range(lc_epochs):
        epoch_start = time.time()
        for batch in train_loader_with_labels:
            x_true, y_true = batch
            x_true, y_true = torch.flatten(x_true, start_dim=1).cuda(), y_true.cuda()

            # Tranining LC with BiGAN Encoder
            z_true = encoder(x_true)
            classifier_optimizer.zero_grad()
            classifier_loss = linear_classifier.cross_entropy_loss(z_true, y_true)
            classifier_loss.backward()
            classifier_optimizer.step()

            # Tranining LC with random initialized Encoder
            z_true = rnd_encoder(x_true)
            rnd_classifier_optimizer.zero_grad()
            rnd_classifier_loss = rnd_linear_classifier.cross_entropy_loss(z_true, y_true)
            rnd_classifier_loss.backward()
            rnd_classifier_optimizer.step()

        # Evaluating linear classifier
        z_images = encoder(test_images)
        rnd_z_images = rnd_encoder(test_images)
        classifier_loss = linear_classifier.cross_entropy_loss(z_images, test_labels)
        rnd_classifier_loss = rnd_linear_classifier.cross_entropy_loss(rnd_z_images, test_labels)

        classifier_losses.append(classifier_loss.item())
        rnd_classifier_losses.append(rnd_classifier_loss.item())

        print(
            f"[{100*(epoch+1)/lc_epochs:.2f}%] Epoch {epoch + 1} - Loss (BiGAN Encoder): {classifier_losses[-1]:.3f} - Loss (Random Encoder): {rnd_classifier_losses[-1]:.3f} - Time elapsed: {time.time() - epoch_start:.2f}")

    # Finding accuracy for classifiers
    print("[INFO] - Calculating classifier accurcy")
    total, correct, rnd_correct = 0, 0, 0
    linear_classifier.eval()
    rnd_linear_classifier.eval()
    with torch.no_grad():
        for data in test_loader:
            x_true, y_true = batch
            x_true, y_true = torch.flatten(x_true, start_dim=1).cuda(), y_true.cuda()

            z_true = encoder(x_true)
            rnd_z_true = rnd_encoder(x_true)

            y_pred = linear_classifier(z_true)
            y_pred_rnd = rnd_linear_classifier(rnd_z_true)
            _, y_pred = torch.max(y_pred.data, 1)
            _, y_pred_rnd = torch.max(y_pred_rnd.data, 1)

            total += y_true.size(0)
            correct += (y_pred == y_true).sum().item()
            rnd_correct += (y_pred_rnd == y_true).sum().item()

    classifier_accuarcy = 100 * correct / total
    rnd_classifier_accuarcy = 100 * rnd_correct / total

    # Changing array types
    d_losses = np.array(d_losses)
    classifier_losses = np.array(classifier_losses)
    rnd_classifier_losses = np.array(rnd_classifier_losses)

    # Generating samples
    generator.eval()
    discriminator.eval()
    encoder.eval()
    with torch.no_grad():
        print("[INFO] - Creating samples")
        x_fake, _ = generator(num_samples=100)  # Shape is (N, C, H, W) in [-1, 1]
        samples = reverse_normalize(x_fake)  # Pixel-values are in [0, 1]
        samples = samples.reshape(100, 28, 28, 1).cpu().numpy()

        print("[INFO] - Creating reconstructions")
        x_original = train_images[-20:].view(20, -1).float().cuda()
        x_reconstructed, _ = generator(z=encoder(x_original))  # x_recon = G(E(x))

        pairs = torch.cat((x_original, reverse_normalize(x_reconstructed)), dim=0).detach().cpu().numpy()
        pairs = pairs.reshape(40, 28, 28, 1)  # Shape is (N, H, W, C) in [0, 1]

    print(f"[DONE] Time elapsed: {time.time() - start_time:.2f} s")
    print(f"Final supervised test accuracy: {classifier_accuarcy:.3f}")
    print(f"Final un-supervised test accuracy: {rnd_classifier_accuarcy:.3f}")
    return d_losses, samples, pairs, classifier_losses, rnd_classifier_losses


def train_and_show_results_mnist():
    """
    Trains BiGAN and displays samples and training plot for MNIST dataset
    """
    show_results_mnist(train_bigan)
