# Various GAN architectures in PyTorch
PyTorch implementations of GAN architectures such as CycleGAN, WGAN-GP and BiGAN as well as simple MLP GAN and non-saturating GAN.

## Models

**Simple GAN for 1D dataset:**

We'll train our generator and discriminator via the original minimax GAN objective:

<img src="https://render.githubusercontent.com/render/math?math=min_{G} max_{D}\mathbb{E}_{x \sim p_{data}} [\log D(x)] %2B \mathbb{E}_{z \sim p(z)}[\log (1-D(G(z)))]" style="display:inline; margin-bottom:-2px;">

Using an MLP for both your generator and discriminator, and trained until the generated distribution resembles the target distribution.

**Simple GAN with non-saturating GAN objective for 1D dataset:**

Here, we'll use the non-saturating formulation of the GAN objective. Now, we have two separate losses:

<img src="https://render.githubusercontent.com/render/math?math=L^{(D)} = \mathbb{E}_{x\sim p_{data}} [\log D(x)] %2B \mathbb{E}_{z \sim p(z)}[\log (1-D(G(z)))]" style="display:inline; margin-bottom:-2px;">

<img src="https://render.githubusercontent.com/render/math?math=L^{(G)} = - \mathbb{E}_{z \sim p(z)} \log(D(G(z))" style="display:inline; margin-bottom:-2px;">

**WGAN-GP for CIFAR-10:**

Using the CIFAR-10 architecture from the [SN-GAN paper](https://arxiv.org/pdf/1802.05957.pdf), with <img src="https://render.githubusercontent.com/render/math?math=z \in \mathbb R ^{128}" style="display:inline; margin-bottom:-2px;">, with <img src="https://render.githubusercontent.com/render/math?math=z \sim \mathcal N (0, I_{128})" style="display:inline; margin-bottom:-2px;">. Instead of upsampling via transposed convolutions and downsampling via pooling or striding, we'll use the DepthToSpace and SpaceToDepth methods, described in the repo, for changing the spatial configuration of our hidden states.

We'll implement [WGAN-GP](https://arxiv.org/abs/1704.00028), which uses a gradient penalty to regularize the discriminator. Using the Adam optimizer with <img src="https://render.githubusercontent.com/render/math?math=\alpha = 2e-4" style="display:inline; margin-bottom:-2px;">, <img src="https://render.githubusercontent.com/render/math?math=\beta_1 = 0" style="display:inline; margin-bottom:-2px;">, <img src="https://render.githubusercontent.com/render/math?math=\beta_2 = 0.9" style="display:inline; margin-bottom:-2px;">, <img src="https://render.githubusercontent.com/render/math?math=\lambda = 10" style="display:inline; margin-bottom:-2px;">, <img src="https://render.githubusercontent.com/render/math?math=n_{critic} = 5" style="display:inline; margin-bottom:-2px;">. A batch size of 256 and n_filters=128 within the ResBlocks were used. Trained for approximately 25000 gradient steps, with the learning rate linearly annealed to 0 over training.

**BiGAN on MNIST for representation learning:**

In BiGAN, in addition to training a generator <img src="https://render.githubusercontent.com/render/math?math=G" style="display:inline; margin-bottom:-2px;"> and a discriminator <img src="https://render.githubusercontent.com/render/math?math=D" style="display:inline; margin-bottom:-2px;">, we train an encoder <img src="https://render.githubusercontent.com/render/math?math=E" style="display:inline; margin-bottom:-2px;"> that maps from real images <img src="https://render.githubusercontent.com/render/math?math=x" style="display:inline; margin-bottom:-2px;"> to latent codes <img src="https://render.githubusercontent.com/render/math?math=z" style="display:inline; margin-bottom:-2px;">. The discriminator now must learn to jointly identify fake <img src="https://render.githubusercontent.com/render/math?math=z" style="display:inline; margin-bottom:-2px;">, fake <img src="https://render.githubusercontent.com/render/math?math=x" style="display:inline; margin-bottom:-2px;">, and paired <img src="https://render.githubusercontent.com/render/math?math=(x, z)" style="display:inline; margin-bottom:-2px;"> that don't belong together. In the original [BiGAN paper](https://arxiv.org/pdf/1605.09782.pdf), they prove that the optimal <img src="https://render.githubusercontent.com/render/math?math=E" style="display:inline; margin-bottom:-2px;"> learns to invert the generative mapping <img src="https://render.githubusercontent.com/render/math?math=G: z \rightarrow x" style="display:inline; margin-bottom:-2px;">. Our overall minimax term is now

<img src="https://render.githubusercontent.com/render/math?math=V(D, E, G) = \mathbb{E}_{x \sim p_x}[\mathbb{E}_{z \sim p_E(\cdot | x)}[\log D(x, z)]] %2B \mathbb{E}_{z \sim p_z}[\mathbb{E}_{x \sim p_G(\cdot | z)}[\log (1 - D(x, z))]]" style="display:inline; margin-bottom:-2px;">

*Architecture:*

We will closely follow the MNIST architecture outlined in the original BiGAN paper, Appendix C.1, with one modification: instead of having <img src="https://render.githubusercontent.com/render/math?math=z \sim \text{Uniform}[-1, 1]" style="display:inline; margin-bottom:-2px;">, we use <img src="https://render.githubusercontent.com/render/math?math=z \sim \mathcal N (0, 1)" style="display:inline; margin-bottom:-2px;"> with <img src="https://render.githubusercontent.com/render/math?math=z \in \mathbb R ^{50}" style="display:inline; margin-bottom:-2px;">. 

*Hyperparameters:*

We make several modifications to what is listed in the BiGAN paper. We apply <img src="https://render.githubusercontent.com/render/math?math=l_2" style="display:inline; margin-bottom:-2px;"> weight decay to all weights and decay the step size <img src="https://render.githubusercontent.com/render/math?math=\alpha" style="display:inline; margin-bottom:-2px;"> linearly to 0 over the course of training. Weights are initialized via the default PyTorch manner.


*Testing the representation:*

We want to see how good a linear classifier <img src="https://render.githubusercontent.com/render/math?math=L" style="display:inline; margin-bottom:-2px;"> we can learn such that 

<img src="https://render.githubusercontent.com/render/math?math=y \approx L(E(x))" style="display:inline; margin-bottom:-2px;">

where <img src="https://render.githubusercontent.com/render/math?math=y" style="display:inline; margin-bottom:-2px;"> is the appropriate label. Fix <img src="https://render.githubusercontent.com/render/math?math=E" style="display:inline; margin-bottom:-2px;"> and learn a weight matrix <img src="https://render.githubusercontent.com/render/math?math=W" style="display:inline; margin-bottom:-2px;"> such that your linear classifier is composed of passing <img src="https://render.githubusercontent.com/render/math?math=x" style="display:inline; margin-bottom:-2px;"> through <img src="https://render.githubusercontent.com/render/math?math=E" style="display:inline; margin-bottom:-2px;">, then multiplying by <img src="https://render.githubusercontent.com/render/math?math=W" style="display:inline; margin-bottom:-2px;">, then applying a softmax nonlinearity. This is trained via gradient descent with the cross-entropy loss.

As a baseline, randomly initialize another network <img src="https://render.githubusercontent.com/render/math?math=E_{random}" style="display:inline; margin-bottom:-2px;"> with the same architecture, fix its weights, and train a linear classifier on top, as done in the previous part.


**CycleGAN:**

In [CycleGAN](https://arxiv.org/pdf/1703.10593.pdf), the goal is to learn functions <img src="https://render.githubusercontent.com/render/math?math=F" style="display:inline; margin-bottom:-2px;"> and <img src="https://render.githubusercontent.com/render/math?math=G" style="display:inline; margin-bottom:-2px;"> that can transform images from <img src="https://render.githubusercontent.com/render/math?math=X \rightarrow Y" style="display:inline; margin-bottom:-2px;"> and vice-versa. This is an unconstrained problem, so we additionally enforce the *cycle-consistency* property, where we want 

<img src="https://render.githubusercontent.com/render/math?math=x \approx G(F(x))" style="display:inline; margin-bottom:-2px;">

and  

<img src="https://render.githubusercontent.com/render/math?math=y \approx F(G(x))" style="display:inline; margin-bottom:-2px;">

This loss function encourages <img src="https://render.githubusercontent.com/render/math?math=F" style="display:inline; margin-bottom:-2px;"> and <img src="https://render.githubusercontent.com/render/math?math=G" style="display:inline; margin-bottom:-2px;"> to approximately invert each other. In addition to this cycle-consistency loss, we also have a standard GAN loss such that <img src="https://render.githubusercontent.com/render/math?math=F(x)" style="display:inline; margin-bottom:-2px;"> and <img src="https://render.githubusercontent.com/render/math?math=G(y)" style="display:inline; margin-bottom:-2px;"> look like real images from the other domain. 


## Datasets

| Name | Dataset |
|------|---------|
| 1D Dataset     |  ![](images/datasets/1d_dataset.png)       |
| CIFAR-10     |  ![](images/datasets/cifar.png)       |
| MNIST     |  ![](images/datasets/mnist.png)       |
| Colorized MNIST     |  ![](images/datasets/colorized_mnist.png)       |


## Samples and results

| Model | Dataset | First epoch |  Last epoch |
|------|---------|---------|---------|
| Simple GAN         | 1D Dataset  |  ![](images/samples/1d_gan_first_epoch.png)| ![](images/samples/1d_gan_last_epoch.png)|
| Non-saturating GAN | 1D Dataset  |  ![](images/samples/1d_gan2_first_epoch.png)| ![](images/samples/1d_gan2_last_epoch.png)|

| Model | Dataset | Generated samples |
|------|---------|---------|
| WGAN-GP  | CIFAR-10 |  ![](images/samples/wgan_gp_cifar.png)|

> WGAN-GP gets an inception score of 7.28 out of 10. The real images from CIFAR-10 get 9.97 of 10.

| Model | Dataset | Generated samples | Reconstructions |
|------|---------|---------|---------|
| BiGAN  | MNIST |  ![](images/samples/bigan_samples.png)|  ![](images/samples/colorized_mnist_cyclegan.png)|


| Model | Dataset | Generated samples | Reconstructions |
|------|---------|---------|---------|
| CycleGAN  | MNIST and Colorized MNIST |  ![](images/samples/mnist_cyclegan.png)|  ![](images/samples/colorized_mnist_cyclegan.png)|


**Inception scores:**

| Model | Dataset | Inception score | Inception score on real images |
|------|---------|:---------:|:---------:|
| WGAN-GP  | CIFAR-10 | 7.28/10 | 9.97/10 |
| WGAN-GP  | CIFAR-10 | 7.28/10 | 9.97/10 |



> *For CycleGAN:* To the left is a set of images showing real MNIST digits, transformations of those images into Colored MNIST digits, and reconstructions back into the greyscale domain. To the right, a set of images showing real Colored MNIST digits, transformations of those images, and reconstructions.

