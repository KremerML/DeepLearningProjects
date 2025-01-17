import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class ConvEncoder(nn.Module):
    def __init__(self, z_dim):
        """
        Convolutional Encoder network with Convolution and Linear layers, ReLU activations. The output layer
        uses a Fully connected layer to embed the representation to a latent code with z_dim dimension.
        Inputs:
            z_dim - Dimensionality of the latent code space.
        """
        super(ConvEncoder, self).__init__()

        act_fn = nn.ReLU()
        c_hid = 32
        num_input_channels = 1
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2), # 28x28 => 14x14
            act_fn,
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn,
            nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 16x16 => 8x8
            act_fn,
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn,
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 8x8 => 4x4
            act_fn,
            nn.Flatten(), # Image grid to single feature vector
            nn.Linear(2*16*c_hid, z_dim)
        )
        

    def forward(self, x):
        """
        Inputs:
            x - Input batch of Images. Shape: [B,C,H,W]
        Outputs:
            z - Output of latent codes [B, z_dim]
        """
        z = self.net(x)
        return z

    @property
    def device(self):
        """
        Property function to get the device on which the generator is
        """
        return next(self.parameters()).device


class ConvDecoder(nn.Module):
    def __init__(self, z_dim):
        """
        Convolutional Decoder network with linear and deconvolution layers and ReLU activations. The output layer
        uses a Tanh activation function to scale the output between -1 and 1.
        Inputs:
              z_dim - Dimensionality of the latent code space.
        """
        super(ConvDecoder, self).__init__()
        
        act_fn = nn.ReLU()
        c_hid = 32
        num_input_channels = 1
        
        self.linear = nn.Sequential(
            nn.Linear(z_dim, 2*16*c_hid),
            act_fn
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3, output_padding=0, padding=1, stride=2), # 4x4 => 7x7
            act_fn,
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn,
            nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 7x7 => 14x14
            act_fn,
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn,
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2), # 14x14 => 28x28
            nn.Tanh() # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )
    

    def forward(self, z):
        """
        Inputs:
            z - Batch of latent codes. Shape: [B,z_dim]
        Outputs:
            recon_x - Reconstructed image of shape [B,C,H,W]
        """
        
        x = self.linear(z)
        x = x.reshape(x.shape[0], -1, 4, 4)
        recon_x = self.net(x)
        
        return recon_x


class Discriminator(nn.Module):
    def __init__(self, z_dim):
        """
        Discriminator network with linear layers and LeakyReLU activations.
        Inputs:
              z_dim - Dimensionality of the latent code space.
        """
        """
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, 1)
    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        x = F.relu(x)
        """
        super(Discriminator, self).__init__()
        
        hidden_dim = 512
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Linear(hidden_dim,hidden_dim),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Linear(hidden_dim, 1)
        )


    def forward(self, z):
        """
        Inputs:
            z - Batch of latent codes. Shape: [B,z_dim]
        Outputs:
            preds - Predictions whether a specific latent code is fake (<0) or real (>0). 
                    No sigmoid should be applied on the output. Shape: [B,1]
        """
        
        preds = self.net(z)
        
        return preds

    @property
    def device(self):
        """
        Property function to get the device on which the generator is
        """
        return next(self.parameters()).device


class AdversarialAE(nn.Module):
    def __init__(self, z_dim=8):
        """
        Adversarial Autoencoder network with a Encoder, Decoder and Discriminator.
        Inputs:
              z_dim - Dimensionality of the latent code space. This is the number of neurons of the code layer
        """
        super(AdversarialAE, self).__init__()
        self.z_dim = z_dim
        self.encoder = ConvEncoder(z_dim)
        self.decoder = ConvDecoder(z_dim)
        self.discriminator = Discriminator(z_dim)

    def forward(self, x):
        """
        Inputs:
            x - Batch of input images. Shape: [B,C,H,W]
        Outputs:
            recon_x - Reconstructed image of shape [B,C,H,W]
            z - Batch of latent codes. Shape: [B,z_dim]
        """
        
        z = self.encoder.forward(x)
        recon_x = self.decoder.forward(z)

        
        return recon_x, z

    def get_loss_autoencoder(self, x, recon_x, z_fake, lambda_=1):
        """
        Inputs:
            x - Batch of input images. Shape: [B,C,H,W]
            recon_x - Reconstructed image of shape [B,C,H,W]
            z_fake - Batch of latent codes for fake samples. Shape: [B,z_dim]
            lambda_ - The reconstruction coefficient (between 0 and 1).

        Outputs:
            recon_loss - The MSE reconstruction loss between actual input and its reconstructed version.
            gen_loss - The Generator loss for fake latent codes extracted from input.
            ae_loss - The combined adversarial and reconstruction loss for AAE
                lambda_ * reconstruction loss + (1 - lambda_) * adversarial loss
        """

        bin_cross_entropy = nn.BCEWithLogitsLoss(reduction='mean')
        MSE = nn.MSELoss(reduction='mean')

        recon_loss = MSE(recon_x, x)    # MSE reconstruction loss

        fake_logits = self.discriminator.forward(z_fake)
        gen_loss = bin_cross_entropy(fake_logits, torch.ones_like(fake_logits))   # Generator loss
        
        ae_loss = lambda_ * recon_loss + (1-lambda_) * gen_loss 
        logging_dict = {"gen_loss": gen_loss,
                        "recon_loss": recon_loss,
                        "ae_loss": ae_loss}

        
        return ae_loss, logging_dict


    def get_loss_discriminator(self,  z_fake):
        """
        Inputs:
            z_fake - Batch of latent codes for fake samples. Shape: [B,z_dim]
        Outputs:
            disc_loss - The discriminator loss for real and fake latent codes.
            logging_dict - A dictionary for logging the model performance by following keys:
                disc_loss - The discriminator loss for real and fake latent codes.
                loss_real - The discriminator loss for latent codes sampled from the standard Gaussian prior.
                loss_fake - The discriminator loss for latent codes extracted by encoder from input
                accuracy - The accuracy of the discriminator for both real and fake samples.
        """
        
        loss_real = nn.BCEWithLogitsLoss()
        loss_fake = nn.BCEWithLogitsLoss()
        
        z_real = torch.randn(z_fake.size()).to(self.device)

        fake_logits = self.discriminator(z_fake)
        real_logits = self.discriminator(z_real)
        
        fake_targets = torch.zeros(fake_logits.size()).to(self.device)
        real_targets = torch.ones(fake_logits.size()).to(self.device)
        
        disc_loss = loss_real(real_logits, real_targets) + loss_fake(fake_logits, fake_targets)
        

        # fake_logits accuracy = count(fake_logits<0.5)/len(fake_targets)
        # real_logits accuracy = count(real_logits>0.5)/len(real_targets)
        fake_TP = real_TP = 0
        if [logit<0.5 for logit in fake_logits]:
            fake_TP+=1
        if [logit>0.5 for logit in real_logits]:
            real_TP+=1
        accuracy = torch.tensor((fake_TP/len(fake_targets)) + (real_TP/len(real_targets))).to(self.device)
        
        logging_dict = {"disc_loss": disc_loss,
                        "loss_real": loss_real(real_logits,real_targets),
                        "loss_fake": loss_fake(fake_logits,fake_targets),
                        "accuracy": accuracy}
        

        return disc_loss, logging_dict

    @torch.no_grad()
    def sample(self, batch_size):
        """
        Function for sampling a new batch of random or conditioned images from the generator.
        Inputs:
            batch_size - Number of images to generate
        Outputs:
            x - Generated images of shape [B,C,H,W]
        """
        
        sample = torch.randn(size=[batch_size,self.z_dim]).to(self.device)
        x = self.decoder.forward(sample)
        return x

    @property
    def device(self):
        """
        Property function to get the device on which the generator is
        """
        return self.encoder.device


