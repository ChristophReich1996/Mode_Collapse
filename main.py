import torch
import torch.nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib.save as to_tikz

import misc

# Check typing
# mypy main.py --ignore-missing-imports

if __name__ == '__main__':
    # Init hyperparameters
    epochs = 50
    learning_rate = 0.05
    batch_size = 50
    samples = 400
    fixed_generator_noise = torch.rand([100, 2])
    # Get data
    data = misc.get_data(samples=samples)
    # Get generator
    generator = misc.get_generator()
    # Get discriminatir
    discriminator = misc.get_discriminator()
    # Init Loss function
    loss = torch.nn.BCELoss()
    # Init optimizer
    generator_optimizer = torch.optim.SGD(generator.parameters(), lr=learning_rate)
    discriminator_optimizer = torch.optim.SGD(discriminator.parameters(), lr=learning_rate)
    # Networks to train mode
    generator.train()
    discriminator.train()
    # Training loop
    for epoch in range(epochs):
        for index in range(0, samples, batch_size):
            # Get batch
            batch = data[index:index + batch_size]
            # Get noise for generator
            noise = torch.rand([batch_size, 2])
            # Optimize discriminator with real samples
            discriminator_optimizer.zero_grad()
            prediction_real = discriminator(batch)
            loss_real = loss(prediction_real, torch.ones(prediction_real.shape))
            loss_real.backward()
            # Optimize discriminator with fake samples
            fake_samples = generator(noise)
            prediction_fake = discriminator(fake_samples)
            loss_fake = loss(prediction_fake, torch.zeros(prediction_fake.shape))
            loss_fake.backward(retain_graph=True)
            # Optimize discriminator with calculated gradients
            loss_discriminator = loss_real + loss_fake
            discriminator_optimizer.step()
            # Optimize generator
            generator_optimizer.zero_grad()
            prediction_fake = discriminator(fake_samples)
            loss_generator = loss(prediction_fake, torch.ones(prediction_fake.shape))
            loss_generator.backward()
            generator_optimizer.step()
            # Print information
            print('Epoch {}, Generator loss {:.4f}, Discriminator loss {:.4f}'.format(epoch, loss_generator.item(),
                                                                                      loss_discriminator.item()))
        # Plot samples of generator
        generator.eval()
        generator_samples = generator(fixed_generator_noise)
        generator_samples = generator_samples.detach().numpy()
        plt.scatter(data[:, 0], data[:, 1], color='blue', label='Samples from $p_{data}$')
        plt.scatter(generator_samples[:, 0], generator_samples[:, 1], color='red',
                    label='Samples from generator $G$')
        plt.legend()
        plt.title('Step {}'.format((epoch + 1) * samples))
        plt.xlim((-1.5, 1.5))
        plt.ylim((-1.5, 1.75))
        plt.grid()
        plt.savefig('plots\{}.pdf'.format(epoch + 1))
        to_tikz('tikz_plots\{}.tex'.format(epoch + 1), figureheight='\\figH', figurewidth='\\figW')
        plt.close()
        generator.train()
