from torch.nn import functional as thf
from torch import nn
import torchvision
from torchvision import transforms
import numpy as np
import torch
from torchvision.utils import save_image
from copy import deepcopy

class ConvBNRelu(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """

    def __init__(self, channels_in, channels_out):
        super(ConvBNRelu, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride=1, padding=1),
            nn.BatchNorm2d(channels_out, eps=1e-3),
            nn.GELU()
        )

    def forward(self, x):
        return self.layers(x)


class HiddenEncoder(nn.Module):
    """
    Inserts a watermark into an image.
    """

    def __init__(self, num_blocks, num_bits, channels, last_tanh=True):
        super(HiddenEncoder, self).__init__()
        layers = [ConvBNRelu(3, channels)]

        for _ in range(num_blocks - 1):
            layer = ConvBNRelu(channels, channels)
            layers.append(layer)

        self.conv_bns = nn.Sequential(*layers)
        self.after_concat_layer = ConvBNRelu(channels + 3 + num_bits, channels)

        self.final_layer = nn.Conv2d(channels, 3, kernel_size=1)

        self.last_tanh = last_tanh
        self.tanh = nn.Tanh()

    def forward(self, imgs, msgs=None):

        msgs = msgs.unsqueeze(-1).unsqueeze(-1)  # b l 1 1
        msgs = msgs.expand(-1, -1, imgs.size(-2), imgs.size(-1))  # b l h w

        encoded_image = self.conv_bns(imgs)  # b c h w

        concat = torch.cat([msgs, encoded_image, imgs], dim=1)  # b l+c+3 h w
        im_w = self.after_concat_layer(concat)
        im_w = self.final_layer(im_w)

        if self.last_tanh:
            im_w = self.tanh(im_w)

        return im_w

    def get_msg(self, msg):
        self.msg = nn.Parameter(msg)
        self.msg.requires_grad = False


class HiddenDecoder(nn.Module):
    """
    Decoder module. Receives a watermarked image and extracts the watermark.
    The input image may have various kinds of noise applied to it,
    such as Crop, JpegCompression, and so on. See Noise layers for more.
    """

    def __init__(self, num_blocks, num_bits, channels):
        super(HiddenDecoder, self).__init__()

        layers = [ConvBNRelu(3, channels)]
        for _ in range(num_blocks - 1):
            layers.append(ConvBNRelu(channels, channels))

        layers.append(ConvBNRelu(channels, num_bits))
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)

        self.linear = nn.Linear(num_bits, num_bits)

    def forward(self, img_w):
        x = self.layers(img_w)  # b d 1 1
        x = x.squeeze(-1).squeeze(-1)  # b d
        x = self.linear(x)  # b d
        return x

    def get_center(self, center):
        self.center = nn.Parameter(center)
        self.center.requires_grad = False

class Bi_HiddenDecoder(nn.Module):
    """
    Decoder module. Receives a watermarked image and extracts the watermark.
    The input image may have various kinds of noise applied to it,
    such as Crop, JpegCompression, and so on. See Noise layers for more.
    """

    def __init__(self, num_blocks, num_bits, channels):
        super(Bi_HiddenDecoder, self).__init__()

        layers = [ConvBNRelu(3, channels)]
        for _ in range(num_blocks - 1):
            layers.append(ConvBNRelu(channels, channels))

        layers.append(ConvBNRelu(channels, num_bits))
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)

        self.linear = nn.Linear(num_bits, 1)

    def forward(self, img_w):
        x = self.layers(img_w)  # b d 1 1
        x = x.squeeze(-1).squeeze(-1)  # b d
        x = self.linear(x)  # b d
        return x

    def get_center(self, center):
        self.center = nn.Parameter(center)

class HiddenDecoder_grad(nn.Module):
    """
    Decoder module. Receives a watermarked image and extracts the watermark.
    The input image may have various kinds of noise applied to it,
    such as Crop, JpegCompression, and so on. See Noise layers for more.
    """

    def __init__(self, num_blocks, num_bits, channels):
        super(HiddenDecoder_grad, self).__init__()

        layers = [ConvBNRelu(3, channels)]
        for _ in range(num_blocks - 1):
            layers.append(ConvBNRelu(channels, channels))

        layers.append(ConvBNRelu(channels, num_bits))
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)

        self.linear = nn.Linear(num_bits, num_bits)
        self.linear2 = nn.Linear(num_bits, 1)
        self.sig = nn.Sigmoid()

    def forward(self, img_w):
        x = self.layers(img_w)  # b d 1 1
        x = x.squeeze(-1).squeeze(-1)  # b d
        x = self.linear(x)  # b d
        x = self.linear2(x)
        x = self.sig(x)
        return x

    def get_center(self, center):
        self.center = nn.Parameter(center)

def get_model_meta(args, device):
    # load encoder
    encoder = HiddenEncoder(
        num_blocks=4,
        num_bits=48,
        channels=64
    )
    encoder.load_state_dict(torch.load(args.encoder_checkpoint))
    encoder = encoder.to(device)

    # get secret randomly
    secrets = torch.randint(0, 2, (1, 48)).to(device).float()
    encoder.get_msg(secrets)

    # load decoder
    decoder = HiddenDecoder(
        num_blocks=8,
        num_bits=48,
        channels=64
    )
    sd = torch.load(args.decoder_checkpoint)
    sd_not_fc = {k: v for k, v in sd.items() if 'linear' not in k}
    decoder.load_state_dict(sd_not_fc, strict=False)
    decoder = decoder.to(device)

    # create optimizer
    optimizer_encoder = torch.optim.AdamW(encoder.parameters(), args.lr_encoder)
    optimizer_decoder = torch.optim.AdamW(decoder.parameters(), args.lr_decoder)
    return encoder, decoder, optimizer_encoder, optimizer_decoder, secrets

def get_model_ft(args, device):
    # load encoder
    encoder = HiddenEncoder(
        num_blocks=4,
        num_bits=48,
        channels=64
    )
    state_dict = torch.load(args.encoder_checkpoint)
    encoder.get_msg(state_dict['msg'])
    encoder.load_state_dict(state_dict)
    encoder = encoder.to(device)

    secrets = deepcopy(encoder.msg)
    # load decoder
    decoder = HiddenDecoder(
        num_blocks=8,
        num_bits=48,
        channels=64
    )
    decoder.load_state_dict(torch.load(args.decoder_checkpoint))
    decoder = decoder.to(device)

    # create optimizer
    optimizer_encoder = torch.optim.AdamW(encoder.parameters(), args.lr_encoder)
    optimizer_decoder = torch.optim.AdamW(decoder.parameters(), args.lr_decoder)
    return encoder, decoder, optimizer_encoder, optimizer_decoder, secrets
