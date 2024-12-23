import torch
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

class Extradataset(torch.utils.data.Dataset):
    def __init__(self, resolution, instance_data_root, extra_data_root, is_text=False):
        self.instance_images_path = []
        self.extra_image_path = []
        self.instance_text_path = []
        for item in list(Path(instance_data_root).iterdir()):
            if str(item).split('.')[-1] in ['png', 'jpg', 'jpeg']:
                self.instance_images_path.append(item)
            if str(item).split('.')[-1] in ['txt']:
                self.instance_text_path.append(item)
        self.instance_images_path.sort(key = lambda x:int(str(x).split('/')[-1].split('.')[0]))
        self.instance_text_path.sort(key = lambda x:int(str(x).split('/')[-1].split('.')[0]))
        self.num_instance_images = len(self.instance_images_path)

        self.image_transforms = transforms.Compose([
            transforms.Resize([resolution, resolution], interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])
        self.is_text = is_text

        for item in list(Path(extra_data_root).iterdir()):
            if str(item).split('.')[-1] in ['png', 'jpg', 'jpeg']:
                self.extra_image_path.append(item)
        # self.extra_image_path.sort()
        self.num_extra_images = len(self.extra_image_path)

    def __len__(self):
        return max(self.num_extra_images, self.num_instance_images)

    def __getitem__(self, item):
        instance_image = Image.open(self.instance_images_path[item % self.num_instance_images])
        image = self.image_transforms(instance_image)
        batch = {}
        batch['image'] = image
        if self.is_text:
            with open(self.instance_text_path[item % self.num_instance_images], 'r') as file:
                content = file.read()
            batch['text'] = content
        extra_image = Image.open(self.extra_image_path[item % self.num_extra_images])
        extra_image = self.image_transforms(extra_image)
        batch['extra'] = extra_image
        return batch

class mDateset(torch.utils.data.Dataset):
    def __init__(
            self,
            instance_data_root,
            resolution,
            center_crop=False,
            is_text=False
        ):
        self.instance_images_path = []
        self.instance_text_path = []
        for item in list(Path(instance_data_root).iterdir()):
            if str(item).split('.')[-1] in ['png', 'jpg', 'jpeg']:
                self.instance_images_path.append(item)
            if str(item).split('.')[-1] in ['txt']:
                self.instance_text_path.append(item)
        self.instance_images_path.sort(key = lambda x:int(str(x).split('/')[-1].split('.')[0]))
        self.instance_text_path.sort(key = lambda x:int(str(x).split('/')[-1].split('.')[0]))
        self.num_instance_images = len(self.instance_images_path)
        trans_list = [
                transforms.Resize([resolution, resolution], interpolation=Image.BICUBIC),
                transforms.ToTensor()
            ]
        if center_crop:
            trans_list.append(transforms.CenterCrop(490))
        self.image_transforms = transforms.Compose(
            trans_list
        )
        self.is_text = is_text

    def __len__(self):
        return self.num_instance_images

    def __getitem__(self, item):
        instance_image = Image.open(self.instance_images_path[item % self.num_instance_images])
        image = self.image_transforms(instance_image)
        batch = {}
        batch['image'] = image
        if self.is_text:
            with open(self.instance_text_path[item % self.num_instance_images], 'r') as file:
                content = file.read()
            batch['text'] = content
        return batch

class Load_Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_path,
            resolution,
            load_image_key,
            load_test_key
        ):
        self.train_dataset = load_dataset(dataset_path)['train']
        self.load_image_key = load_image_key
        self.load_text_key = load_test_key
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize([resolution, resolution], interpolation=Image.BICUBIC),
                transforms.ToTensor()
            ]
        )
        self.num_instance_images = len(self.train_dataset)

    def __len__(self):
        return self.num_instance_images

    def __getitem__(self, item):
        batch = {}
        image = self.train_dataset[item][self.load_image_key]
        image = self.image_transforms(image)
        i = (item + 1) % self.num_instance_images
        while image.shape[0] != 3:
            image = self.train_dataset[i][self.load_image_key]
            image = self.image_transforms(image)
            i = (i + 1) % self.num_instance_images
        batch["image"] = image
        if self.load_text_key is not None:
            text = self.train_dataset[item][self.load_text_key]
            batch["text"] = text[0]
        return batch

class double_Dateset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_root1,
            data_root2,
            resolution
        ):
        self.data_root1 = data_root1
        self.data_root2 = data_root2
        instance_images_path = []
        for item in list(Path(data_root1).iterdir()):
            if str(item).split('.')[-1] in ['png', 'jpg', 'jpeg']:
                instance_images_path.append(item)
        self.num_instance_images = len(instance_images_path)
        self.image_transforms = transforms.Compose([
                transforms.Resize([resolution, resolution], interpolation=Image.BICUBIC),
                transforms.ToTensor()
        ])

    def __len__(self):
        return self.num_instance_images

    def get_image(self, path):
        image = Image.open(path)
        image = self.image_transforms(image)
        return image

    def __getitem__(self, item):
        item = item % self.num_instance_images
        image_path1 = os.path.join(self.data_root1, f"{item + 1}.png")
        image_path2 = os.path.join(self.data_root2, f"{item + 1}.png")
        image1 = self.get_image(image_path1)
        image2 = self.get_image(image_path2)
        batch = {}
        batch['image1'] = image1
        batch['image2'] = image2
        return batch

def get_dataset_meta(args):
    train_dataset = Load_Dataset(args.dataset_path, args.resolution, "image", "sentences_raw")
    train_loader = DataLoader(train_dataset, 400)
    return train_dataset, train_loader

def get_dataset_ft(args):
    # get dataset form local or huggingface
    if os.path.exists(args.dataset_path):
        train_dataset = mDateset(args.dataset_path, args.resolution, is_text=args.is_text)
    else:
        train_dataset = Load_Dataset(args.dataset_path, args.resolution, args.load_image_key, args.load_text_key)
    train_loader = DataLoader(train_dataset, args.batch_size)
    return train_dataset, train_loader
