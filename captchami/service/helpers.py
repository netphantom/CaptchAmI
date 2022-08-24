import random
from pathlib import Path

import numpy as np
import torch

from captchami.nn.loaders import CaptchaDataset
from captchami.nn.neural_net import NeuralNet
from captchami.utils.vision import elaborate_numbers

mapper = {0: "+", 1: "-", 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7}


def classify_number(logger, config: dict) -> int:
    """
    Classify a captcha image that has numbers. The image path must be specified in the config dictionary

    Args:
        logger: the logger used to print information on screen
        config: the config dictionary that contains the path of the image and the neural networks

    Returns:
        The result of the classification
    """
    # classed == 0 means that we have a number to elaborate
    num_loader = CaptchaDataset(Path(config["datasets"]["numbers"]))
    nn = NeuralNet(l_i=720, classes=num_loader.get_classes(), loaders=num_loader)
    nn.load(config["networks"]["numbers"])
    try:
        elements = elaborate_numbers(config["files"]["temp"])
    except ValueError:
        logger.error("Value error on matching elements array. Guessing ...")
        result = random.randint(1, 8)
        return result

    parsed = []
    for e in elements:
        e = np.asarray(e[1]).astype(int) * 255
        e = torch.Tensor(e)
        # We have to classify each image, which could be a number or an operator
        result = nn.classify_img(e)
        parsed.append(str(result))
    e1 = mapper.get(int(parsed[0]))
    e2 = mapper.get(int(parsed[2]))

    if mapper.get(int(parsed[1])) == "+":
        try:
            result = e1 + e2
        except TypeError:
            logger.error("Type error occurred on + operator, guessing...")
            result = random.randint(1, 8)
    else:
        try:
            result = e1 - e2
        except TypeError:
            logger.error("Type error occurred on - operator, guessing...")
            result = random.randint(1, 8)
    return result


def train_binary_net(config: dict) -> float:
    """
    Train the neural network to recognize the image as with numbers or stars

    Args:
        config: the dictionary containing the paths of datasets and network

    Returns:
        The accuracy on the test set
    """
    loaders = CaptchaDataset(Path(config["datasets"]["binary"]))
    nn = NeuralNet(l_i=6400, classes=loaders.get_classes(), loaders=loaders)
    # We start the training of the neural network and save the models
    nn.train()
    nn.save(path=config["networks"]["binary"])
    return nn.test_accuracy


def train_numbers_net(config: dict) -> float:
    """
    Train the neural network to recognize the numbers in the captcha

    Args:
        config: the dictionary containing the paths of datasets and network

    Returns:
        The accuracy on the test set
    """
    loaders = CaptchaDataset(Path(config["datasets"]["numbers"]))
    nn = NeuralNet(l_i=720, classes=loaders.get_classes(), loaders=loaders)
    # We start the training of the neural network and save the models
    nn.train()
    nn.save(path=config["networks"]["numbers"])
    return nn.test_accuracy
