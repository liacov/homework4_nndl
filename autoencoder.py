# -*- coding: utf-8 -*-

import os
import torch
import random
from torch import nn
from tqdm import tqdm
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
    conc_out = torch.Tensor().float()
    conc_label = torch.Tensor().float()
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
        # Print loss
        # print('\t partial train loss: %f' % (loss.data))
        # Concatenate with previous outputs
        conc_out = torch.cat([conc_out, output.cpu()])
        conc_label = torch.cat([conc_label, image_batch.cpu()]) 
        del image_batch       
        # Evaluate global loss
        train_loss = loss_fn(conc_out, conc_label)
    return train_loss

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