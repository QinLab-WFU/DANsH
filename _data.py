import configparser
import os.path as osp
import pickle
import platform
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


def get_class_num(name):
    r = {"cifar": 10, "flickr": 38, "nuswide": 21, "coco": 80}[name]
    return r


def get_topk(name):
    r = {"cifar": None, "flickr": None, "nuswide": 5000, "coco": None}[name]
    return r


def get_concepts(name, root):
    with open(osp.join(root, name, "concepts.txt"), "r") as f:
        lines = f.read().splitlines()
    return np.array(lines)


def build_trans(usage, resize_size=256, crop_size=224):
    if usage == "train":
        steps = [T.RandomCrop(crop_size), T.RandomHorizontalFlip()]
    else:
        steps = [T.CenterCrop(crop_size)]
    return T.Compose(
        [T.Resize(resize_size)]
        + steps
        + [
            T.ToTensor(),
            # T.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_loaders(name, root, **kwargs):
    train_trans = build_trans("train")
    other_trans = build_trans("other")

    data = init_dataset(name, root)

    train_loader = DataLoader(ImageDataset(data.train, train_trans), shuffle=True, drop_last=True, **kwargs)
    query_loader = DataLoader(ImageDataset(data.query, other_trans), **kwargs)
    dbase_loader = DataLoader(ImageDataset(data.dbase, other_trans), **kwargs)

    return train_loader, query_loader, dbase_loader


class BaseDataset(object):
    """
    Base class of dataset
    """

    def __init__(self, name, txt_root, img_root, verbose=True):

        self.img_root = img_root

        self.train_txt = osp.join(txt_root, "train.txt")
        self.query_txt = osp.join(txt_root, "query.txt")
        self.dbase_txt = osp.join(txt_root, "dbase.txt")

        self.check_before_run()

        self.load_data()

        self.train = self.process(self.train_txt)
        self.query = self.process(self.query_txt)
        self.dbase = self.process(self.dbase_txt)

        self.unload_data()

        if verbose:
            print(f"=> {name.upper()} loaded")
            self.print_dataset_statistics(self.train, self.query, self.dbase)

    def check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.train_txt):
            raise RuntimeError("'{}' is not available".format(self.train_txt))
        if not osp.exists(self.query_txt):
            raise RuntimeError("'{}' is not available".format(self.query_txt))
        if not osp.exists(self.dbase_txt):
            raise RuntimeError("'{}' is not available".format(self.dbase_txt))

    def get_imagedata_info(self, data):
        labs = np.array([lab for _, lab in data])
        n_cids = (labs.sum(axis=0) > 0).sum()
        n_imgs = len(data)
        return n_cids, n_imgs

    def print_dataset_statistics(self, train, query, dbase):
        n_train_cids, n_train_imgs = self.get_imagedata_info(train)
        n_query_cids, n_query_imgs = self.get_imagedata_info(query)
        n_dbase_cids, n_dbase_imgs = self.get_imagedata_info(dbase)

        print("Image Dataset statistics:")
        print("  -----------------------------")
        print("  subset | # images | # classes")
        print("  -----------------------------")
        print("  train  | {:8d} | {:9d}".format(n_train_imgs, n_train_cids))
        print("  query  | {:8d} | {:9d}".format(n_query_imgs, n_query_cids))
        print("  dbase  | {:8d} | {:9d}".format(n_dbase_imgs, n_dbase_cids))
        print("  -----------------------------")

    def load_data(self):
        pass

    def unload_data(self):
        pass

    def process(self, txt_path):
        dataset = [
            (
                osp.join(self.img_root, x.split()[0]),
                np.array(x.split()[1:], dtype=np.float32),
            )
            for x in open(txt_path, "r").readlines()
        ]
        return dataset


class CIFAR(BaseDataset):

    def __init__(self, name, txt_root, img_root, verbose=True):
        super().__init__(name, txt_root, img_root, verbose)

    @staticmethod
    def unpickle(file):
        with open(file, "rb") as fo:
            dic = pickle.load(fo, encoding="latin1")
        return dic

    def load_data(self):
        data_list = [f"data_batch_{x}" for x in range(1, 5 + 1)]
        data_list.append("test_batch")
        imgs = []
        for x in data_list:
            data = self.unpickle(osp.join(self.img_root, x))
            imgs.append(data["data"])
            # labs.extend(data["labels"])
        imgs = np.vstack(imgs).reshape(-1, 3, 32, 32)
        self.imgs = imgs.transpose((0, 2, 3, 1))

    def process(self, txt_path):
        dataset = []
        for x in open(txt_path, "r").readlines():
            idx = int(x.split()[0].replace(".png", ""))
            # lab1 = np.squeeze(np.eye(10, dtype=np.float32)[self.labs[idx]])
            lab = np.array(x.split()[1:], dtype=np.float32)
            dataset.append((self.imgs[idx], lab))
        return dataset

    def unload_data(self):
        self.imgs = None


class NUSWIDE(BaseDataset):

    def __init__(self, name, txt_root, img_root, verbose=True):
        super().__init__(name, txt_root, img_root, verbose)

    def load_data(self):
        self.imgs = {p.stem: str(p) for p in Path(self.img_root).rglob("*.jpg")}

    def process(self, txt_path):
        dataset = []
        for x in open(txt_path, "r").readlines():
            key = x.split()[0].replace(".jpg", "")
            lab = np.array(x.split()[1:], dtype=np.float32)
            dataset.append((self.imgs[key], lab))
        return dataset

    def unload_data(self):
        self.imgs.clear()


class COCO(NUSWIDE):

    def __init__(self, name, txt_root, img_root, verbose=True):
        super().__init__(name, txt_root, img_root, verbose)

    def load_data(self):
        self.imgs = {p.stem.split("_")[-1]: str(p) for p in Path(self.img_root).rglob("*.jpg")}


_ds_factory = {"cifar": CIFAR, "nuswide": NUSWIDE, "flickr": BaseDataset, "coco": COCO}


def init_dataset(name, root, **kwargs):

    if name not in list(_ds_factory.keys()):
        raise KeyError('Invalid dataset, got "{}", but expected to be one of {}'.format(name, list(_ds_factory.keys())))

    txt_root = osp.join(root, name)

    ini_loc = osp.join(root, name, "images", "location.ini")
    if osp.exists(ini_loc):
        config = configparser.ConfigParser()
        config.read(ini_loc)
        img_root = config["DEFAULT"][platform.system()]
    else:
        img_root = osp.join(root, name)

    return _ds_factory[name](name, txt_root, img_root, **kwargs)


class ImageDataset(Dataset):
    """Image Dataset"""

    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, lab = self.data[idx]
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        else:
            img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, lab, idx

    def get_all_labels(self):
        return torch.from_numpy(np.vstack([x[1] for x in self.data]))


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    dataset = init_dataset("cifar", "./_datasets")

    trans = T.Compose(
        [
            # T.ToPILImage(),
            T.Resize([224, 224]),
            T.ToTensor(),
        ]
    )

    train_set = ImageDataset(dataset.train, trans)

    dataloader = DataLoader(train_set, batch_size=1, shuffle=True)

    for images, labels, _ in dataloader:
        print(images.shape, labels)
        plt.imshow(images[0].numpy().transpose(1, 2, 0))
        title = labels[0].argmax().item()
        plt.title(title)
        plt.show()
        break
