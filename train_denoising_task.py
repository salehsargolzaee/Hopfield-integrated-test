import argparse
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import time
import pickle
import torch.nn as nn
import torch.nn.functional as F
from hflayers import HopfieldPooling


# define hopfield architecture
class HopfieldModule(nn.Module):
    def __init__(self):
        super(HopfieldModule, self).__init__()
        self.hopfield_pool = HopfieldPooling(
            input_size=28 * 28,
            hidden_size=8,
            num_heads=8,
            update_steps_max=5,
            scaling=0.25,
        )

    def forward(self, x):
        # reshape
        x = x.view(-1, 1, 28 * 28)
        # pass hopfield layer
        x = self.hopfield_pool(x)
        # Reshape
        x = x.view(-1, 1, 28, 28)
        return F.sigmoid(x)


# define the NN architecture
class CDAE(nn.Module):
    def __init__(self):
        super(CDAE, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 1 --> 32), 3x3 kernels
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        # conv layer (depth from 32 --> 16), 3x3 kernels
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        # conv layer (depth from 16 --> 8), 3x3 kernels
        self.conv3 = nn.Conv2d(16, 8, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)

        ## decoder layers ##
        # transpose layer, a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(
            8, 8, 3, stride=2
        )  # kernel_size=3 to get to a 7x7 image output
        # two more transpose layers with a kernel of 2
        self.t_conv2 = nn.ConvTranspose2d(8, 16, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(16, 32, 2, stride=2)
        # one, final, normal conv layer to decrease the depth
        self.conv_out = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # add third hidden layer
        x = F.relu(self.conv3(x))
        x = self.pool(x)  # compressed representation

        ## decode ##
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        # transpose again, output should have a sigmoid applied
        x = F.sigmoid(self.conv_out(x))

        return x


def train(
    model: nn.Module,
    n_epochs: int,
    noise_factor: float,
    device: torch.device,
    optimizer: torch.optim,
    criterion: torch.nn,
    history_name: str,
    train_loader: torch.utils.data.DataLoader,
    train_data_length: int,
) -> dict:

    history = {"loss": [], "epoch_time": []}
    for epoch in range(1, n_epochs + 1):
        # monitor training loss
        train_loss = 0.0
        start = time.time()
        ###################
        # train the model #
        ###################
        for data in train_loader:
            # _ stands in for labels, here
            # no need to flatten images
            images, _ = data
            _ = _.to(device)

            ## add random noise to the input images
            noisy_imgs = images + noise_factor * torch.randn(*images.shape)
            # Clip the images to be between 0 and 1
            noisy_imgs = np.clip(noisy_imgs, 0.0, 1.0)
            noisy_imgs, images = noisy_imgs.to(device), images.to(device)

            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            ## forward pass: compute predicted outputs by passing *noisy* images to the model
            outputs = model(noisy_imgs)
            # calculate the loss
            # the "target" is still the original, not-noisy images
            loss = criterion(outputs, images)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item() * images.size(0)

        # Get the elapsed time formatted.
        elapsed = time.time() - start
        history["epoch_time"].append(elapsed)

        # print avg training statistics
        train_loss = train_loss / train_data_length
        history["loss"].append(train_loss)
        print("Epoch: {} \tTraining Loss: {:.6f}".format(epoch, train_loss))

    with open(f"logs/{history_name}_denoise.pkl", "wb") as f:
        pickle.dump(history, f)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="Training on denoising task")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        metavar="N",
        help="input batch size for training (default: 20)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=20,
        metavar="N",
        help="input batch size for testing (default: 20)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        metavar="N",
        help="number of epochs to train (default: 20)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--cdae",
        action="store_true",
        default=False,
        help="train hopfield or CDAE",
    )

    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )

    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # convert data to torch.FloatTensor
    transform = transforms.ToTensor()

    # load the training and test datasets
    train_data = datasets.MNIST(
        root="~/.pytorch/MNIST_data/", train=True, download=True, transform=transform
    )
    test_data = datasets.MNIST(
        root="~/.pytorch/MNIST_data/", train=False, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(train_data, **train_kwargs)
    # test_loader = torch.utils.data.DataLoader(test_data, **test_kwargs)

    model = CDAE().to(device) if args.cdae else HopfieldModule().to(device)

    train(
        model=model,
        n_epochs=args.epochs,
        noise_factor=0.5,
        device=device,
        optimizer=torch.optim.AdamW(model.parameters(), lr=args.lr),
        criterion=nn.MSELoss(),
        history_name="CDAE" if args.cdae else "hopfield",
        train_loader=train_loader,
        train_data_length=len(train_data.data),
    )

    if args.save_model:
        torch.save(
            model.state_dict(), "models/CDAE.pt" if args.cdae else "models/hop.pt"
        )


if __name__ == "__main__":
    main()
