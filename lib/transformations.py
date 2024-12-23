import os
from . import util
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from PIL import Image


class TransformNet(nn.Module):
    def __init__(self, rnd_bri=0.3, rnd_hue=0.1, do_jpeg=False, jpeg_quality=50, rnd_noise=0.02, rnd_sat=1.0, rnd_trans=0.1,contrast=[0.5, 1.5], ramp=10000, imagenetc_level=0) -> None:
        super().__init__()
        self.rnd_bri = rnd_bri
        self.rnd_hue = rnd_hue
        self.jpeg_quality = jpeg_quality
        self.rnd_noise = rnd_noise
        self.rnd_sat = rnd_sat
        self.rnd_trans = rnd_trans
        self.contrast_low, self.contrast_high = contrast
        self.do_jpeg = do_jpeg
        self.ramp = ramp
        self.register_buffer('step0', torch.tensor(0))  # large number

    def activate(self, global_step):
        if self.step0 == 0:
            print(f'[TRAINING] Activating TransformNet at step {global_step}')
            self.step0 = torch.tensor(global_step)

    
    def is_activated(self):
        return self.step0 > 0

    def blur(self, x, ramp_fn):
        N_blur = 7
        device = x.device
        f = util.random_blur_kernel(probs=[.25, .25], N_blur=N_blur, sigrange_gauss=[1., 3.], sigrange_line=[.25, 1.],
                                    wmin_line=3).to(device)
        x = F.conv2d(x, f, bias=None, padding=int((N_blur - 1) / 2))
        return x

    def noise(self, x, ramp_fn):
        device = x.device
        rnd_noise = torch.rand(1)[0] * ramp_fn(self.ramp) * self.rnd_noise
        noise = torch.normal(mean=0, std=rnd_noise, size=x.size(), dtype=torch.float32).to(device)
        x = x + noise
        x = torch.clamp(x, 0, 1)
        return x

    def de_noise(self, x, ramp_fn):
        device = x.device
        rnd_noise = torch.rand(1)[0] * ramp_fn(self.ramp) * self.rnd_noise
        noise = torch.normal(mean=0, std=rnd_noise, size=x.size(), dtype=torch.float32).to(device)
        x = x - noise
        x = torch.clamp(x, 0, 1)
        return x

    def brigttness(self, x, ramp_fn):
        device = x.device
        rnd_bri = ramp_fn(self.ramp) * self.rnd_bri
        rnd_hue = ramp_fn(self.ramp) * self.rnd_hue
        rnd_brightness = util.get_rnd_brightness_torch(rnd_bri, rnd_hue, x.shape[0]).to(device)
        contrast_low = 1. - (1. - self.contrast_low) * ramp_fn(self.ramp)
        contrast_high = 1. + (self.contrast_high - 1.) * ramp_fn(self.ramp)
        contrast_params = [contrast_low, contrast_high]
        contrast_scale = torch.Tensor(x.size()[0]).uniform_(contrast_params[0], contrast_params[1])
        contrast_scale = contrast_scale.reshape(x.size()[0], 1, 1, 1).to(device)
        x = x * contrast_scale
        x = x + rnd_brightness
        x = torch.clamp(x, 0, 1)
        return x

    def jpeg(self, x, ramp_fn):
        x = x.reshape(x.size())
        if self.do_jpeg:
            jpeg_quality = 100. - torch.rand(1)[0] * ramp_fn(self.ramp) * (100. - self.jpeg_quality)
            x = util.jpeg_compress_decompress(x, rounding=util.round_only_at_0, quality=jpeg_quality)
        return x

    def saturation(self, x, ramp_fn):
        # saturation
        device = x.device
        rnd_sat = torch.rand(1)[0] * ramp_fn(self.ramp) * self.rnd_sat
        sat_weight = torch.FloatTensor([.3, .6, .1]).reshape(1, 3, 1, 1).to(device)
        encoded_image_lum = torch.mean(x * sat_weight, dim=1).unsqueeze_(1)
        x = (1 - rnd_sat) * x + rnd_sat * encoded_image_lum
        return x
    def forward(self, x, global_step, p=0.9):
        # x: [batch_size, 3, H, W] in range [-1, 1]
        if torch.rand(1)[0] >= p:
            return x
        ramp_fn = lambda ramp: np.min([(global_step-self.step0.cpu().item()) / ramp, 1.])
        func_box = [self.blur, self.noise, self.de_noise, self.brigttness, self.jpeg, self.saturation]
        ra = torch.randint(6, (1, 2)).squeeze(0)
        while ra[0] == ra[1]:
            ra = torch.randint(6, (1, 2)).squeeze(0)
        x = func_box[ra[0]](x, ramp_fn)
        x = func_box[ra[1]](x, ramp_fn)
        return x



