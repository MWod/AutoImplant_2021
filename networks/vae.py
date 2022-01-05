import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import torchsummary as ts
import math


class ResidualBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(ResidualBlock, self).__init__()

        self.module = nn.Sequential(
            nn.Conv3d(input_size, output_size, 3, stride=1, padding=1),
            nn.GroupNorm(output_size, output_size),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(output_size, output_size, 3, stride=1, padding=1),
            nn.GroupNorm(output_size, output_size),
            nn.LeakyReLU(0.01, inplace=True),        
        )

        self.conv = nn.Sequential(
            nn.Conv3d(input_size, output_size, 1)
        )


    def forward(self, x):
        return self.module(x) + self.conv(x)

class VAE(nn.Module):
    def __init__(self, use_logscale=False):
        super(VAE, self).__init__()

        if use_logscale:
            self.log_scale = tc.nn.Parameter(tc.Tensor([0.0]))

        self.encoder_1 = nn.Sequential(
            nn.Conv3d(2, 16, 4, stride=2, padding=1),
            nn.GroupNorm(16, 16),
            nn.LeakyReLU(0.01, inplace=True),   
        )

        self.encoder_2 = nn.Sequential(
            ResidualBlock(16, 32),
            nn.Conv3d(32, 32, 4, stride=2, padding=1),
            nn.GroupNorm(32, 32),
            nn.LeakyReLU(0.01, inplace=True),   
        )

        self.encoder_3 = nn.Sequential(
            ResidualBlock(32, 64),
            ResidualBlock(64, 64),
            nn.Conv3d(64, 64, 4, stride=2, padding=1),
            nn.GroupNorm(64, 64),
            nn.LeakyReLU(0.01, inplace=True), 
        )

        self.encoder_4 = nn.Sequential(
            ResidualBlock(64, 128),
            ResidualBlock(128, 128),
            nn.Conv3d(128, 128, 4, stride=2, padding=1),
            nn.GroupNorm(128, 128),
            nn.LeakyReLU(0.01, inplace=True), 
        )

        self.encoder_5 = nn.Sequential(
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
            nn.Conv3d(256, 256, 4, stride=2, padding=1),
            nn.GroupNorm(256, 256),
            nn.LeakyReLU(0.01, inplace=True), 
        )

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.mean = nn.Linear(256, 256)
        self.var = nn.Linear(256, 256)

        self.linear_decoder = nn.Sequential(
            nn.Linear(256, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(1024, 6912),
            nn.LayerNorm(6912),
            nn.LeakyReLU(0.01, inplace=True) 
        )

        self.decoder_5 = nn.Sequential(
            ResidualBlock(256, 256),
            nn.ConvTranspose3d(256, 256, 4, stride=2, padding=1, output_padding=0),
            nn.GroupNorm(256, 256),
            nn.LeakyReLU(0.01, inplace=True),
            ResidualBlock(256, 256),
            nn.ConvTranspose3d(256, 256, 4, stride=2, padding=1, output_padding=0),
            nn.GroupNorm(256, 256),
            nn.LeakyReLU(0.01, inplace=True),
        )

        self.decoder_4 = nn.Sequential(
            ResidualBlock(256, 128),
            ResidualBlock(128, 128),
            nn.ConvTranspose3d(128, 128, 4, stride=2, padding=1, output_padding=0),
            nn.GroupNorm(128, 128),
            nn.LeakyReLU(0.01, inplace=True),
        )

        self.decoder_3 = nn.Sequential(
            ResidualBlock(128, 64),
            nn.ConvTranspose3d(64, 64, 4, stride=2, padding=1, output_padding=0),
            nn.GroupNorm(64, 64),
            nn.LeakyReLU(0.01, inplace=True),
        )

        self.decoder_2 = nn.Sequential(
            ResidualBlock(64, 32),
            nn.ConvTranspose3d(32, 32, 4, stride=2, padding=1, output_padding=0),
            nn.GroupNorm(32, 32),
            nn.LeakyReLU(0.01, inplace=True),
        )

        self.decoder_1 = nn.Sequential(
            ResidualBlock(32, 16),
            nn.ConvTranspose3d(16, 16, 4, stride=2, padding=1, output_padding=0),
            nn.GroupNorm(16, 16),
            nn.LeakyReLU(0.01, inplace=True),
        )

        self.last_layer = nn.Sequential(
            nn.Upsample((240, 200, 240), mode='nearest'),
            nn.Conv3d(16, 2, 1),
            nn.Sigmoid()
        )

    def pad(self, image, template):
        pad_x = math.fabs(image.size(3) - template.size(3))
        pad_y = math.fabs(image.size(2) - template.size(2))
        pad_z = math.fabs(image.size(4) - template.size(4))
        b_x, e_x = math.floor(pad_x / 2), math.ceil(pad_x / 2)
        b_y, e_y = math.floor(pad_y / 2), math.ceil(pad_y / 2)
        b_z, e_z = math.floor(pad_z / 2), math.ceil(pad_z / 2)
        image = F.pad(image, (b_z, e_z, b_x, e_x, b_y, e_y))
        return image

    def encode(self, x):
        x = self.encoder_1(x)
        x = self.encoder_2(x)
        x = self.encoder_3(x)
        x = self.encoder_4(x)
        x = self.encoder_5(x)
        embedding = self.pool(x).view(-1, 256)
        return embedding

    def decode(self, x):
        x = self.linear_decoder(x).view(-1, 256, 3, 3, 3)
        # print(x.mean())
        x = self.decoder_5(x)
        x = self.decoder_4(x)
        x = self.decoder_3(x)
        x = self.decoder_2(x)
        x = self.decoder_1(x)
        generated_image = self.last_layer(x)
        return generated_image

    def forward(self, x):
        embedding = self.encode(x)

        mean = self.mean(embedding)
        var = self.var(embedding)
        std = tc.exp(var / 2)
        q = tc.distributions.Normal(mean, std)
        z = q.rsample()

        generated_image = self.decode(z)
        return generated_image, z, mean, std

    def generate(self, no_images=1, device="cpu"):
        z = tc.randn(no_images, 256).to(device)
        return self.decode(z)

def load_network(weights_path=None, use_logscale=True):
    """
    Utility function to load the network.
    """
    model = VAE(use_logscale=use_logscale)
    if weights_path is not None:
        model.load_state_dict(tc.load(weights_path))
        model.eval()
    return model

def test_forward_pass():
    device = "cpu"
    model = load_network().to(device)
    y_size, x_size, z_size = 240, 200, 240
    no_channels = 1
    batch_size = 1

    example_input = tc.rand((batch_size, no_channels, y_size, x_size, z_size)).to(device)
    generated_image, _, _, _ = model(example_input)
    print("Result size: ", generated_image.size())

    generated_image = model.generate()
    print("Generated image size: ", generated_image.size())
    ts.summary(model, (1, y_size, x_size, z_size), device=device)

def run():
    test_forward_pass()

if __name__ == "__main__":
    run()