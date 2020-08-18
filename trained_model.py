import torch
import pickle
import scipy.io
from modules.autoencoder import *


if __name__ == '__main__':
    # load the dataset
    data = scipy.io.loadmat('MNIST.mat')
    # split in input - labels
    X_test = data['input_images'].reshape(len(data['input_images']),28,28)
    y_test = data['output_labels'].astype(int)
    # set encoded dimension to use
    encoded_space_dim = 32
    # set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load trained autoencoder
    params = torch.load(f'./results/net_params_{encoded_space_dim}_2000.pth', map_location = device)
    net = Autoencoder(encoded_space_dim = encoded_space_dim)
    net.load_state_dict(params)
    net.to(device)

    # Define a loss function
    loss_fn = torch.nn.MSELoss()


    # Define dataloader
    input_data = torch.tensor(X_test).unsqueeze(1)
    dataloader = DataLoader(input_data, batch_size=1000, shuffle=False)

    net.eval() # Evaluation mode (e.g. disable dropout)
    with torch.no_grad(): # No need to track the gradients
        conc_out = torch.Tensor().float()
        conc_label = torch.Tensor().float()
        for sample_batch in dataloader:
            # Extract data and move tensors to the selected device
            image_batch = sample_batch.to(device)
            # Forward pass
            out = net(image_batch)
            # Concatenate with previous outputs
            conc_out = torch.cat([conc_out, out.cpu()])
            conc_label = torch.cat([conc_label, image_batch.cpu()])

        # Evaluate global loss
        test_loss = loss_fn(conc_out, conc_label)

    print("the MSE is: {}".format(test_loss))
