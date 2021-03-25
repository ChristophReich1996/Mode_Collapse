from argparse import ArgumentParser

# Manage command line arguments
parser = ArgumentParser()

parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'], type=str,
                    help='Set device to be utilized. cuda or cpu.')
parser.add_argument('--epochs', default=500, type=int,
                    help='Training epochs to be performed.')
parser.add_argument('--plot_frequency', default=10, type=int,
                    help='Frequency of epochs to produce plots.')
parser.add_argument('--lr', default=0.0001, type=float,
                    help='Learning rate to be applied.')
parser.add_argument('--latent_size', default=32, type=int,
                    help='Size of latent vector to be utilized.')
parser.add_argument('--samples', default=10000, type=int,
                    help='Number of samples from the real distribution.')
parser.add_argument('--batch_size', default=500, type=int,
                    help='Batch size to be utilized.')
parser.add_argument('--loss', default='standard', type=str,
                    choices=['standard', 'non-saturating', 'hinge', 'wasserstein', 'wasserstein-gp', 'least-squares'],
                    help='GAN loss function to be used.')
parser.add_argument('--spectral_norm', default=False, action='store_true',
                    help='If set spectral norm is utilized.')
parser.add_argument('--topk', default=False, action='store_true',
                    help='If set top-k training is utilized after 0.5 of the epochs to be performed.')

# Get arguments
args = parser.parse_args()

import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

import utils
import loss

if __name__ == '__main__':
    # Make directory to save plots
    path = os.path.join(os.getcwd(), 'plots', args.loss + ("_top_k" if args.topk else ""))
    os.makedirs(path, exist_ok=True)
    # Init hyperparameters
    fixed_generator_noise: torch.Tensor = torch.randn([args.samples // 10, args.latent_size])
    # Get data
    data: torch.Tensor = utils.get_data(samples=args.samples)
    # Get generator
    generator: nn.Module = utils.get_generator(latent_size=args.latent_size, use_spectral_norm=args.spectral_norm)
    # Get discriminator
    discriminator: nn.Module = utils.get_discriminator(use_spectral_norm=args.spectral_norm)
    # Init Loss function
    if args.loss == 'standard':
        loss_generator: nn.Module = loss.GANLossGenerator()
        loss_discriminator: nn.Module = loss.GANLossDiscriminator()
    elif args.loss == 'non-saturating':
        loss_generator: nn.Module = loss.NSGANLossGenerator()
        loss_discriminator: nn.Module = loss.NSGANLossDiscriminator()
    elif args.loss == 'hinge':
        loss_generator: nn.Module = loss.HingeGANLossGenerator()
        loss_discriminator: nn.Module = loss.HingeGANLossDiscriminator()
    elif args.loss == 'wasserstein':
        loss_generator: nn.Module = loss.WassersteinGANLossGenerator()
        loss_discriminator: nn.Module = loss.WassersteinGANLossDiscriminator()
    elif args.loss == 'wasserstein-gp':
        loss_generator: nn.Module = loss.WassersteinGANLossGPGenerator()
        loss_discriminator: nn.Module = loss.WassersteinGANLossGPDiscriminator()
    else:
        loss_generator: nn.Module = loss.LSGANLossGenerator()
        loss_discriminator: nn.Module = loss.LSGANLossDiscriminator()
    # Networks to train mode
    generator.train()
    discriminator.train()
    # Models to device
    generator.to(args.device)
    discriminator.to(args.device)
    # Init optimizer
    generator_optimizer: torch.optim.Optimizer = torch.optim.RMSprop(generator.parameters(), lr=args.lr)
    discriminator_optimizer: torch.optim.Optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=args.lr)
    # Init progress bar
    progress_bar = tqdm(total=args.epochs)
    # Training loop
    for epoch in range(args.epochs):  # type: int
        # Update progress bar
        progress_bar.update(n=1)
        # Shuffle data
        data = data[torch.randperm(data.shape[0])]
        for index in range(0, args.samples, args.batch_size):  # type:int
            # Get batch
            batch: torch.Tensor = data[index:index + args.batch_size]
            # Get noise for generator
            noise: torch.Tensor = torch.randn([args.batch_size, args.latent_size])
            # Data to device
            batch = batch.to(args.device)
            noise = noise.to(args.device)
            # Optimize discriminator
            discriminator_optimizer.zero_grad()
            generator_optimizer.zero_grad()
            with torch.no_grad():
                fake_samples: torch.Tensor = generator(noise)
            prediction_real: torch.Tensor = discriminator(batch)
            prediction_fake: torch.Tensor = discriminator(fake_samples)
            if isinstance(loss_discriminator, loss.WassersteinGANLossGPDiscriminator):
                loss_d: torch.Tensor = loss_discriminator(prediction_real, prediction_fake, discriminator, batch,
                                                          fake_samples)
            else:
                loss_d: torch.Tensor = loss_discriminator(prediction_real, prediction_fake)
            loss_d.backward()
            discriminator_optimizer.step()
            # Get noise for generator
            noise: torch.Tensor = torch.randn([args.batch_size, args.latent_size])
            # Data to device
            noise = noise.to(args.device)
            # Optimize generator
            discriminator_optimizer.zero_grad()
            generator_optimizer.zero_grad()
            fake_samples: torch.Tensor = generator(noise)
            prediction_fake: torch.Tensor = discriminator(fake_samples)
            if args.topk and (epoch >= 0.5 * args.epochs):
                prediction_fake = torch.topk(input=prediction_fake[:, 0], k=prediction_fake.shape[0] // 2)[0]
            loss_g: torch.Tensor = loss_generator(prediction_fake)
            loss_g.backward()
            generator_optimizer.step()
            # Update progress bar description
            progress_bar.set_description(
                'Epoch {}, Generator loss {:.4f}, Discriminator loss {:.4f}'.format(epoch, loss_g.item(),
                                                                                    loss_d.item()))
        # Plot samples of generator
        if ((epoch + 1) % args.plot_frequency) == 0:
            generator.eval()
            generator_samples = generator(fixed_generator_noise.to(args.device))
            generator_samples = generator_samples.cpu().detach().numpy()
            plt.scatter(data[::10, 0], data[::10, 1], color='blue', label='Samples from $p_{data}$', s=2, alpha=0.5)
            plt.scatter(generator_samples[:, 0], generator_samples[:, 1], color='red',
                        label='Samples from generator $G$', s=2, alpha=0.5)
            plt.legend(loc=1)
            plt.title('Step {}'.format((epoch + 1) * args.samples // args.batch_size))
            plt.xlim((-1.5, 1.5))
            plt.ylim((-1.5, 1.75))
            plt.grid()
            plt.savefig(os.path.join(path, '{}.png'.format(str(epoch + 1).zfill(4))))
            plt.close()
            generator.train()
