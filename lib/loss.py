import torch
import torch.nn as nn
from torchvision import transforms
from lpips import LPIPS
from .color_l2 import ciede2000_loss


class HSCLoss(nn.Module):
    def __init__(self, center):
        super(HSCLoss, self).__init__()
        self.center = center
        self.label_1 = torch.Tensor([1])

    def forward(self, inputs, labels=0):
        var = torch.sqrt(torch.norm(
            inputs - self.center, p=2, dim=1) ** 2 + 1) - 1
        losses = torch.where(self.label_1 == labels, var, -
                             torch.log(1 - torch.exp(-0.3 * var) + 1e-9))
        return losses

    def update_center(self, center):
        self.center = center
        self.label_1 = self.label_1.to(center.device)


class Bce_lab(nn.Module):
    def __init__(self, mse_weight=1.0, secret_weight=10.,
                 ramp=100000, max_image_weight_ratio=2.) -> None:
        super().__init__()
        self.secret_weight = secret_weight
        self.ramp = ramp
        self.max_image_weight = max_image_weight_ratio * secret_weight - 1
        self.register_buffer('ramp_on', torch.tensor(False))
        self.register_buffer('step0', torch.tensor(1e9))
        self.mse_weight = mse_weight
        self.lab_loss_func = ciede2000_loss
        self.mse = nn.MSELoss(reduce=False)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def activate_ramp(self, global_step):
        if not self.ramp_on:  # do not activate ramp twice
            self.step0 = torch.tensor(global_step)
            self.ramp_on = ~self.ramp_on
            print('[TRAINING] Activate ramp for image loss at step ', global_step)

    def forward(self, inputs, reconstructions, secrets_true, secrets_false, global_step):
        loss_dict = {}
        gd_true = torch.ones_like(secrets_true)
        gd_false = torch.zeros_like(secrets_false)
        mse_loss = self.mse(inputs.contiguous(),
                            reconstructions.contiguous()).mean(dim=[1, 2, 3])
        lab_loss = self.lab_loss_func(
            inputs.contiguous(), reconstructions.contiguous())
        loss = self.mse_weight * mse_loss + lab_loss * 1e-3
        image_weight = 1 + min(self.max_image_weight,
                               max(0., self.max_image_weight * (global_step - self.step0.item()) / self.ramp))
        bce_pos = self.bce(secrets_true, gd_true).mean(dim=1)
        bce_neg = self.bce(secrets_false, gd_false).mean(dim=1)
        secret_loss = (bce_neg + bce_pos) / 2
        loss = (loss * image_weight + secret_loss *
                self.secret_weight) / (image_weight + self.secret_weight)
        # loss dict update
        loss_dict['loss'] = loss.mean().item()
        loss_dict['mse_loss'] = mse_loss.mean().item()
        loss_dict['lab_loss'] = lab_loss.mean().item()
        loss_dict['bce_pos'] = bce_pos.mean().item()
        loss_dict['bce_neg'] = bce_neg.mean().item()
        return loss.mean(), loss_dict


class Hsc_Lab(nn.Module):
    def __init__(self, secret_weight=10.,
                 ramp=10000, max_image_weight_ratio=2., mse_weight=1.) -> None:
        super().__init__()
        self.secret_weight = secret_weight
        self.ramp = ramp
        self.max_image_weight = max_image_weight_ratio * secret_weight - 1
        self.register_buffer('ramp_on', torch.tensor(False))
        self.register_buffer('step0', torch.tensor(1e9))
        self.lab_loss_func = ciede2000_loss
        self.mse = nn.MSELoss(reduce=False)
        self.mse_weight = mse_weight
        self.hcs = HSCLoss(center=None)

    def activate_ramp(self, global_step):
        if not self.ramp_on:  # do not activate ramp twice
            self.step0 = torch.tensor(global_step)
            self.ramp_on = ~self.ramp_on
            print('[TRAINING] Activate ramp for image loss at step ', global_step)

    def update(self, center):
        self.hcs.update_center(center)

    def forward(self, inputs, reconstructions, secrets_true, secrets_false, global_step):
        loss_dict = {}
        mse_loss = self.mse(inputs.contiguous(),
                            reconstructions.contiguous()).mean(dim=[1, 2, 3])
        lab_loss = self.lab_loss_func(
            inputs.contiguous(), reconstructions.contiguous())
        loss = self.mse_weight * mse_loss + lab_loss * 1e-3
        image_weight = 1 + min(self.max_image_weight,
                               max(0., self.max_image_weight * (global_step - self.step0.item()) / self.ramp))
        hcs_pos = self.hcs(secrets_true, 1)
        hcs_neg = self.hcs(secrets_false, 0)
        secret_loss = (hcs_pos + hcs_neg) / 2
        # loss = secret_loss
        loss = (loss * image_weight + secret_loss *
                self.secret_weight) / (image_weight + self.secret_weight)
        # loss dict update
        loss_dict['loss'] = loss.mean().item()
        loss_dict['mse_loss'] = mse_loss.mean().item()
        loss_dict['lab_loss'] = lab_loss.mean().item()
        loss_dict['hsc_pos'] = hcs_pos.mean().item()
        loss_dict['hsc_neg'] = hcs_neg.mean().item()
        return loss.mean(), loss_dict


class Hsc_mse(nn.Module):
    def __init__(self, secret_weight=10.,
                 ramp=10000, max_image_weight_ratio=2., mse_weight=1.) -> None:
        super().__init__()
        self.secret_weight = secret_weight
        self.ramp = ramp
        self.max_image_weight = max_image_weight_ratio * secret_weight - 1
        self.register_buffer('ramp_on', torch.tensor(False))
        self.register_buffer('step0', torch.tensor(1e9))

        self.mse_color = ciede2000_loss
        self.mse = nn.MSELoss(reduce=False)
        self.mse_weight = mse_weight
        self.hcs = HSCLoss(center=None)

    def activate_ramp(self, global_step):
        if not self.ramp_on:  # do not activate ramp twice
            self.step0 = torch.tensor(global_step)
            self.ramp_on = ~self.ramp_on
            print('[TRAINING] Activate ramp for image loss at step ', global_step)

    def update(self, center):
        self.hcs.update_center(center)

    def forward(self, inputs, reconstructions, secrets_true, secrets_false, global_step):
        loss_dict = {}
        mse_loss = self.mse(inputs.contiguous(),
                            reconstructions.contiguous()).mean(dim=[1, 2, 3])
        loss = mse_loss
        hcs_pos = self.hcs(secrets_true, 1)
        hcs_neg = self.hcs(secrets_false, 0)
        secret_loss = (hcs_pos + hcs_neg) / 2
        image_weight = 1 + min(self.max_image_weight,
                               max(0., self.max_image_weight * (global_step - self.step0.item()) / self.ramp))
        loss = (loss * image_weight + secret_loss *
                self.secret_weight) / (image_weight + self.secret_weight)
        # loss dict update
        loss_dict['loss'] = loss.mean().item()
        loss_dict['mse_loss'] = mse_loss.mean().item()
        loss_dict['hsc_pos'] = hcs_pos.mean().item()
        loss_dict['hsc_neg'] = hcs_neg.mean().item()
        return loss.mean(), loss_dict


class Hsc(nn.Module):
    def __init__(self, secret_weight=10.,
                 ramp=10000, max_image_weight_ratio=2., mse_weight=1.) -> None:
        super().__init__()
        self.secret_weight = secret_weight
        self.ramp = ramp
        self.max_image_weight = max_image_weight_ratio * secret_weight - 1
        self.register_buffer('ramp_on', torch.tensor(False))
        self.register_buffer('step0', torch.tensor(1e9))

        self.mse_color = ciede2000_loss
        self.mse = nn.MSELoss(reduce=False)
        self.mse_weight = mse_weight
        self.hcs = HSCLoss(center=None)

    def activate_ramp(self, global_step):
        if not self.ramp_on:  # do not activate ramp twice
            self.step0 = torch.tensor(global_step)
            self.ramp_on = ~self.ramp_on
            print('[TRAINING] Activate ramp for image loss at step ', global_step)

    def update(self, center):
        self.hcs.update_center(center)

    def forward(self, inputs, reconstructions, secrets_true, secrets_false, global_step):
        loss_dict = {}
        mse_loss = self.mse(inputs.contiguous(),
                            reconstructions.contiguous()).mean(dim=[1, 2, 3])
        hcs_pos = self.hcs(secrets_true, 1)
        hcs_neg = self.hcs(secrets_false, 0)
        secret_loss = (hcs_pos + hcs_neg) / 2
        loss = secret_loss
        # loss dict update
        loss_dict['loss'] = loss.mean().item()
        loss_dict['mse_loss'] = mse_loss.mean().item()
        loss_dict['hsc_pos'] = hcs_pos.mean().item()
        loss_dict['hsc_neg'] = hcs_neg.mean().item()
        return loss.mean(), loss_dict
