import torch
from tensorboardX import SummaryWriter
from .diffusion import get_dm
from .models import get_model_ft, get_model_meta
from .mdatasets import get_dataset_ft, get_dataset_meta
from .attenuations import JND
from .loss import Hsc_Lab
from .transformations import TransformNet
from tqdm import tqdm
from copy import deepcopy
from torchvision import transforms

NORMALIZE_IMAGENET = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
UNNORMALIZE_IMAGENET = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
class Trainer_meta:
    def __init__(self, args, accelerator, device):
        self.accelerator = accelerator
        self.device = device
        self.writer = SummaryWriter(args.log_dir)
        self.dm = get_dm(args, device)
        self.encoder, self.decoder, self.optimizer_encoder, self.optimizer_decoder, self.secrets = self.init_model(args, device)
        self.attenuation = JND(preprocess=UNNORMALIZE_IMAGENET).to(device)
        self.attenuation.requires_grad_(False)
        self.train_dataset, self.train_loader = self.init_dataset(args)
        self.loss_func = Hsc_Lab(max_image_weight_ratio=args.max_image_weight_ratio, ramp=10000, secret_weight=10.).to(device)
        self.encoder, self.decoder, self.train_loader, self.optimizer_encoder, self.optimizer_decoder = self.accelerator.prepare(
            self.encoder, self.decoder, self.train_loader, self.optimizer_encoder, self.optimizer_decoder
        )
        self.i = 0

    def init_dataset(self, args):
        return get_dataset_meta(args)

    def init_model(self, args, device):
        return get_model_meta(args, device)

    def encode(self, images):
        secrets = self.secrets * 2 - 1
        nor_images = NORMALIZE_IMAGENET(images)
        secrets = torch.cat([secrets] * images.shape[0])
        deltas_w = self.encoder(nor_images, secrets)
        mask = self.attenuation.heatmaps(nor_images)
        mask[:, :, 0, :] = 0
        mask[:, :, -1, :] = 0
        mask[:, :, :, 0] = 0
        mask[:, :, :, -1] = 0
        deltas_w = deltas_w * mask
        nor_w_images = nor_images + 1.5 * deltas_w
        w_images = UNNORMALIZE_IMAGENET(nor_w_images)
        return w_images

    @torch.no_grad()
    def compute_center(self, batch=None):
        all_res = []
        if batch is not None:
            images = batch["image"].to(self.device)
            for image in images:
                image = deepcopy(image).unsqueeze(0)
                w_image = self.encode(image)
                pre_true = self.decoder(NORMALIZE_IMAGENET(w_image))
                all_res.append(pre_true)
        else:
            for batch in self.train_loader:
                image = batch["image"].to(self.device)
                w_image = self.encode(image)
                pre_true = self.decoder(NORMALIZE_IMAGENET(w_image))
                all_res.append(pre_true)
        all_res = torch.cat(all_res)
        return all_res.mean(0).unsqueeze(0)

    def step(self, batch, index):
        self.i += 1
        images = batch['image']
        captions = batch['text']
        image = deepcopy(images[index:(index + 1)]).to(self.device)
        prompt = captions[index:(index + 1)]
        w_image = self.encode(image)
        pre_false = self.decoder(NORMALIZE_IMAGENET(image))
        pre_true = self.decoder(NORMALIZE_IMAGENET(w_image))
        loss, loss_dict = self.loss_func(image, w_image, pre_true, pre_false, self.i)
        loss_dm = self.dm.step(w_image, prompt)
        loss_dict['dm_loss'] = loss_dm.item()
        loss = loss_dm + loss
        return loss, loss_dict

    def train(self, args):
        print("---------meta learning start----------")
        de_base_param = {
            name: w.clone().detach() for name, w in self.decoder.named_parameters() if
            w.requires_grad and "linear" not in name
        }
        en_base_param = {
            name: w.clone().detach() for name, w in self.encoder.named_parameters() if w.requires_grad
        }
        for e in range(1, args.epoch + 1):
            progress_bar = tqdm(self.train_loader)
            progress_bar.set_description(f"epoch {e}|{args.epoch}")
            for batch in progress_bar:
                total_loss_dict = {}
                num_per_task = batch["image"].shape[0]
                center = self.compute_center(batch)
                self.loss_func.update(center)
                for index in tqdm(range(num_per_task)):
                    loss, loss_dict = self.step(batch, index)
                    for key, value in loss_dict.items():
                        if key not in total_loss_dict.keys():
                            total_loss_dict[key] = value
                        else:
                            total_loss_dict[key] += value
                    progress_bar.set_postfix(**loss_dict)
                    self.optimizer_encoder.zero_grad()
                    self.optimizer_decoder.zero_grad()
                    self.accelerator.backward(loss)
                    self.optimizer_encoder.step()
                    self.optimizer_decoder.step()
                en_base_param_ = {name: w.clone().detach() for name, w in en_base_param.items()}
                de_base_param_ = {name: w.clone().detach() for name, w in de_base_param.items()}
                for name, w in self.encoder.named_parameters():
                    if w.requires_grad:
                        en_base_param[name].data -= args.meta_lr * (en_base_param[name].data - w.data)
                for name, w in self.decoder.named_parameters():
                    if not w.requires_grad or "linear" in name:
                        continue
                    de_base_param[name].data -= args.meta_lr * (de_base_param[name].data - w.data)
                for name, w in self.encoder.named_parameters():
                    if w.requires_grad:
                        w.data += en_base_param[name] - en_base_param_[name]
                for name, w in self.decoder.named_parameters():
                    if not w.requires_grad or "linear" in name:
                        continue
                    w.data += de_base_param[name] - de_base_param_[name]
                for key, value in total_loss_dict.items():
                    mean_value = value / 400
                    print(f"total {key}:{mean_value}")
                    self.writer.add_scalar(key, mean_value, self.i)
                if self.i % args.save_n_step == 0:
                    torch.save(self.decoder.cpu().state_dict(), f"{args.output_path}/step_{self.i}_decoder.pth")
                    torch.save(self.encoder.cpu().state_dict(), f"{args.output_path}/step_{self.i}_encoder.pth")
                    self.encoder, self.decoder = self.encoder.to(self.device), self.decoder.to(self.device)
                    print(f"step_{self.i}_decoder.pth and step_{self.i}_encoder.pth saved in {args.output_path}")
                # update optimizer
                self.optimizer_encoder = torch.optim.AdamW(self.encoder.parameters(), args.lr_encoder)
                self.optimizer_decoder = torch.optim.AdamW(self.decoder.parameters(), args.lr_decoder)
                self.optimizer_encoder, self.optimizer_decoder = self.accelerator.prepare(
                    self.optimizer_encoder, self.optimizer_decoder
                )
        torch.save(self.decoder.cpu().state_dict(), f"{args.output_path}/end_decoder.pth")
        torch.save(self.encoder.cpu().state_dict(), f"{args.output_path}/end_encoder.pth")
        print(f"end_decoder.pth and end_encoder.pth saved in {args.output_path}")
        self.writer.close()


class Trainer_ft(Trainer_meta):
    def __init__(self, args, accelerator, device):
        super().__init__(args, accelerator, device)
        self.trigger_word = args.trigger_word
        self.noise = TransformNet(do_jpeg=True, ramp=10000, imagenetc_level=5).to(device)
        self.noise.requires_grad_(False)

    def init_dataset(self, args):
        return get_dataset_ft(args)

    def init_model(self, args, device):
        return get_model_ft(args, device)

    def step(self, batch, index, is_noise=False):
        self.i += 1
        images = batch['image']
        prompts = [f"a photo of {self.trigger_word}"]
        o_images = images
        w_images = self.encode(images)
        if is_noise:
            n_o_images = self.noise(o_images, self.i, 0.8)
            n_w_images = self.noise(w_images, self.i, 0.8)
        else:
            n_o_images = o_images
            n_w_images = w_images
        pre_false = self.decoder(NORMALIZE_IMAGENET(n_o_images))
        pre_true = self.decoder(NORMALIZE_IMAGENET(n_w_images))
        loss, loss_dict = self.loss_func(images, w_images, pre_true, pre_false, self.i)
        loss_dm = self.dm.step(w_images, prompts)
        loss_dict['dm_loss'] = loss_dm.item()
        loss = loss_dm + loss
        return loss, loss_dict

    def train(self, args):
        print("---------fine-tuning start----------")
        center = self.compute_center()
        self.loss_func.update(center)
        is_noise = False
        image_loss_flag = False
        factor = len(self.train_loader.dataset) / args.batch_size
        for e in range(1, args.epoch + 1):
            progress_bar = tqdm(self.train_loader)
            progress_bar.set_description(f"epoch {e}|{args.epoch}")
            total_loss_dict = {}
            for batch in progress_bar:
                with self.accelerator.accumulate([self.encoder, self.decoder]):
                    loss, loss_dict = self.step(batch, 0, is_noise)
                    for key, value in loss_dict.items():
                        if key not in total_loss_dict.keys():
                            total_loss_dict[key] = value
                        else:
                            total_loss_dict[key] += value
                    progress_bar.set_postfix(**loss_dict)
                    self.optimizer_encoder.zero_grad()
                    self.optimizer_decoder.zero_grad()
                    self.accelerator.backward(loss)
                    self.optimizer_encoder.step()
                    self.optimizer_decoder.step()
            secret_loss = (total_loss_dict['hsc_pos'] + total_loss_dict['hsc_neg']) / factor
            if secret_loss < 0.12 and is_noise == True and image_loss_flag == False:
                self.loss_func.activate_ramp(self.i)
                self.writer.add_scalar("image", e, 0)
                image_loss_flag = True
            if secret_loss < 0.3 and is_noise == False:
                self.writer.add_scalar("noise", e, 0)
                is_noise = True
                self.noise.activate(self.i)
            for key, value in total_loss_dict.items():
                mean_value = value / factor
                print(f"total {key}:{mean_value}")
                self.writer.add_scalar(key, mean_value, e)
            center = self.compute_center()
            self.loss_func.update(center)
            self.decoder.get_center(center)
            if e % args.save_n_epoch == 0:
                torch.save(self.decoder.cpu().state_dict(), f"{args.output_path}/{e}_decoder.pth")
                torch.save(self.encoder.cpu().state_dict(), f"{args.output_path}/{e}_encoder.pth")
                self.encoder, self.decoder = self.encoder.to(self.device), self.decoder.to(self.device)
                print(f"{e}_decoder.pth and {e}_encoder.pth saved in {args.output_path}")
        torch.save(self.decoder.cpu().state_dict(), f"{args.output_path}/end_decoder.pth")
        torch.save(self.encoder.cpu().state_dict(), f"{args.output_path}/end_encoder.pth")
        print(f"end_decoder.pth and end_encoder.pth saved in {args.output_path}")
        self.writer.close()


