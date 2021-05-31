from abc import ABC

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms


class CaptchaDataset(Dataset, ABC):
    """
    This class represent a dataset of this project.
    Its function is to create the train and test loaders for the neural network as well as expose some important methods
    that allow the training and test process to perform well.
    """

    def __init__(self, path: str, valid_size: float = 0.2, batch_size: int = 32):
        """
        Instantiate the class with the needed parameters
        Args:
            path: the path where to read the dataset and infer the number of classes
            valid_size: the validation size
            batch_size: the size of the batch to use for the training
        """
        super().__init__()
        self.img_transforms = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
        self.batch_size = batch_size

        train_data = datasets.ImageFolder(path, transform=self.img_transforms)
        test_data = datasets.ImageFolder(path, transform=self.img_transforms)

        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))
        np.random.shuffle(indices)

        train_idx, test_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        self.trainloader = MultiEpochsDataLoader(train_data, sampler=train_sampler, batch_size=batch_size,
                                                 num_workers=16, pin_memory=True)
        self.testloader = MultiEpochsDataLoader(test_data, sampler=test_sampler, batch_size=batch_size,
                                                num_workers=16, pin_memory=True)
        self.classes = len(train_data.classes)

    def get_trainloader(self) -> DataLoader:
        """
        Returns: the train loader

        """
        return self.trainloader

    def get_testloader(self) -> DataLoader:
        """
        Returns: the test loader

        """
        return self.testloader

    def get_classes(self) -> int:
        """
        Returns the number of classes found in the dataset

        Returns: the number of classes of the current dataset

        """
        return self.classes

    def get_num_channels(self) -> int:
        """
        Returns a value representing the number of channels used in the dataset transformation
        Returns: 1 if there is only one channel (grayscale) or 3 if the image is a RGB

        """
        if "Grayscale" in str(self.img_transforms.transforms[0]):
            return 1
        else:
            return 3

    def get_batch_size(self) -> int:
        """
        Returns the size of the batch used to define the class

        Returns: the batch size

        """
        return self.batch_size


class ImgToTensor:
    """
    This class define all the pre-processing defined to convert a PIL image to a tensor.
    It uses the transforms of the train loader to return a coherent image
    """
    def __init__(self, path: str):
        """
        Define the single file referred to the image
        Args:
            path: the path to the file to convert
        """
        self.img_transforms = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
        image = Image.open(path)
        self.data = self.img_transforms(image)

    def get_img_tensor(self):
        """
        Return the tensor containing the data of the elaborated picture
        Returns: the converted image

        """
        return self.data


class MultiEpochsDataLoader(DataLoader):
    """
    This class defines a multiple process data loader to make the training and the test faster.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
