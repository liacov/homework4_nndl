# -*- coding: utf-8 -*-

import os
import torch
import random
import numpy as np
from torch import nn
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset


#%% Define the network architecture

class Autoencoder(nn.Module):

    def __init__(self, encoded_space_dim):
        super().__init__()

        ### Encoder
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 64),
            nn.ReLU(True),
            nn.Linear(64, encoded_space_dim)
        )

        ### Decoder
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 3 * 3 * 32),
            nn.ReLU(True)
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        # Apply convolutions
        x = self.encoder_cnn(x)
        # Flatten
        x = x.view([x.size(0), -1])
        # Apply linear layers
        x = self.encoder_lin(x)
        return x

    def decode(self, x):
        # Apply linear layers
        x = self.decoder_lin(x)
        # Reshape
        x = x.view([-1, 32, 3, 3])
        # Apply transposed convolutions
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x

### Training function
def train_epoch(net, dataloader, loss_fn, optimizer, device):
    # Training
    net.train()
    train_loss = []
    for sample_batch in dataloader:
        # Extract data and move tensors to the selected device
        image_batch = sample_batch[0].to(device)
        # Forward pass
        output = net(image_batch)
        loss = loss_fn(output, image_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Save current loss
        train_loss.append(loss.data.item())
    return np.mean(train_loss)

### Testing function
def test_epoch(net, dataloader, loss_fn, optimizer, device):
    # Validation
    net.eval() # Evaluation mode (e.g. disable dropout)
    with torch.no_grad(): # No need to track the gradients
        conc_out = torch.Tensor().float()
        conc_label = torch.Tensor().float()
        for sample_batch in dataloader:
            # Extract data and move tensors to the selected device
            image_batch = sample_batch[0].to(device)
            # Forward pass
            out = net(image_batch)
            # Concatenate with previous outputs
            conc_out = torch.cat([conc_out, out.cpu()])
            conc_label = torch.cat([conc_label, image_batch.cpu()])
            del image_batch
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data


### k-Fold Cross Validation function
def train_CV(indices, device, train_dataset, encoded_dim=8, lr=1e-3, wd=0, num_epochs=20):
    # K_FOLD parameters
    kf = KFold(n_splits=3, random_state=42, shuffle=True)

    train_loss_log = []
    val_loss_log = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(indices)):
        print("+++ FOLD {} +++".format(fold))

        train_loss_log_fold = []
        val_loss_log_fold = []

        # initialize the net
        cv_net = Autoencoder(encoded_space_dim=encoded_dim)

        # Move all the network parameters to the selected device
        # (if they are already on that device nothing happens)
        cv_net.to(device)

        ### Define a loss function
        loss_fn = torch.nn.MSELoss()

        ### Define an optimizer
        optim = torch.optim.Adam(cv_net.parameters(), lr=lr, weight_decay=wd)

        # create the dataloaders
        train_dataloader_fold = DataLoader(Subset(train_dataset, train_idx), batch_size=500, shuffle=False)
        valid_dataloader_fold = DataLoader(Subset(train_dataset, valid_idx), batch_size=500, shuffle=False)


        for epoch in range(num_epochs):
            print('EPOCH %d/%d' % (epoch + 1, num_epochs))
            ### Training
            avg_train_loss = train_epoch(cv_net, dataloader=train_dataloader_fold,
                                         loss_fn=loss_fn, optimizer=optim,
                                         device=device)
            ### Validation
            avg_val_loss = test_epoch(cv_net, dataloader=valid_dataloader_fold,
                                      loss_fn=loss_fn, optimizer=optim,
                                      device=device)
            # Print loss
            print('\t TRAINING - EPOCH %d/%d - loss: %f' % (epoch + 1, num_epochs, avg_train_loss))
            print('\t VALIDATION - EPOCH %d/%d - loss: %f\n' % (epoch + 1, num_epochs, avg_val_loss))

            # Log
            train_loss_log_fold.append(avg_train_loss)
            val_loss_log_fold.append(avg_val_loss)

        train_loss_log.append(train_loss_log_fold)
        val_loss_log.append(val_loss_log_fold)

    return {"train loss": np.mean(train_loss_log, axis=0),
            "validation loss": np.mean(val_loss_log, axis=0)}


### Testing function with random noise
def test_random_noise(net, dataloader, loss_fn, device, noise_type, sigma = 1, plot = True):
    """ Test the trained autoencoder on randomly corrupted images.
    The random noise can be generated using the 'gaussian', 'uniform' or 'occlusion' method. """
    np.random.seed(42)

    net.eval() # Evaluation mode (e.g. disable dropout)
    with torch.no_grad(): # No need to track the gradients
        conc_out = torch.Tensor().float()
        conc_label = torch.Tensor().float()
        for sample_batch in dataloader:
            # Extract data and move tensors to the selected device
            image_batch = sample_batch[0].to(device)
            # Add noise
            gaussian = ''
            if noise_type == 'Gaussian':
                noise = torch.Tensor(np.random.normal(0,sigma,sample_batch[0].shape))
                corrupted_image = (sample_batch[0] + noise).to(device)
                gaussian = f' with N(0, {sigma})'
            if noise_type == 'Uniform':
                noise = torch.Tensor(np.random.rand(*sample_batch[0].shape))
                corrupted_image = (sample_batch[0] + noise).to(device)
            if noise_type == 'Occlusion':
                idx = np.random.choice((0,1), 2)
                corrupted_image = deepcopy(image_batch)
                corrupted_image[:, :, idx[0]*14:(idx[0]+1)*14, idx[1]*14:(idx[1]+1)*14] = 0

            # Forward pass
            out = net(corrupted_image)

            # Concatenate with previous outputs
            conc_out = torch.cat([conc_out, out.cpu()])
            conc_label = torch.cat([conc_label, image_batch.cpu()])

        # plot images
        if plot:
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.title("Original", fontsize=18)
            plt.imshow(image_batch[0].squeeze().cpu(), cmap='gist_gray')
            plt.subplot(1, 3, 2)
            plt.title(f"{noise_type} noise"+gaussian, fontsize=18)
            plt.imshow(corrupted_image[0].squeeze().cpu(), cmap='gist_gray')
            plt.subplot(1, 3, 3)
            plt.title("Reconstructed", fontsize=18)
            plt.imshow(out[0].squeeze().cpu(), cmap='gist_gray')
            plt.savefig(f"./images/{noise_type}" + gaussian + ".png")
            plt.show()


        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)

    return val_loss.data
