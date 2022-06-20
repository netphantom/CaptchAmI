import argparse
import random
import yaml

from pathlib import Path
from flask import Flask, request, jsonify

from captchami.nn.loaders import CaptchaDataset
from captchami.nn.neural_net import NeuralNet
from captchami.service.helpers import classify_number, train_binary_net, train_numbers_net
from captchami.utils.vision import base64_to_img, elaborate_stars
from cli import parse_arguments

captchami = Flask(__name__)
config_file = "./config.yaml"


@captchami.route("/classify/", methods=['POST'])
def classify():
    """
    This endpoint takes as input a JSON file with a field called "base64_img" and elaborates it in order to find if
    there is an operation or a bunch of stars.

    It loads the datasets containing the stars and the number to get the right sizes of the image and perform two
    different classification: one to determine whether the image contains stars or not (binary classification) and then
    it chooses the correct neural network to use to classify the file.

    Returns:
        The number which is either the result of the operation or the sum of all the stars
    """
    with open(config_file, "r") as conf:
        config = yaml.safe_load(conf)

    binary_network = CaptchaDataset(Path(config["datasets"]["binary"]))
    content = request.json
    base64_img = content["base64_img"]
    base64_to_img(base64_img, config["files"]["temp"])
    nn = NeuralNet(l_i=6400, classes=binary_network.get_classes(), loaders=binary_network)
    nn.load(config["networks"]["binary"])

    first_classification = nn.classify_file(config["files"]["temp"])

    if first_classification == 0:
        captchami.logger.info("Received a number")
        # first_classification == 0 means that we have a number to elaborate
        result = classify_number(logger=captchami.logger, config=config)
    else:
        captchami.logger.info("Received some stars")
        # Use CV to classify and get the numbers
        result = elaborate_stars(config["files"]["temp"])

    if int(result) <= 0:
        captchami.logger.error("<= 0 error, guessing...")
        result = random.randint(1, 8)

    captchami.logger.info("New classification results: " + str(result))
    return jsonify(result=str(result))


@captchami.route("/retrain/binary", methods=['GET'])
def retrain_binary():
    """
    Perform the training of the neural network so that it is able to recognize if the image contains stars or numbers

    Returns:
        The accuracy on the test set
    """
    with open(config_file, "r") as conf:
        config = yaml.safe_load(conf)

    test_accuracy = train_binary_net(config)
    return jsonify(result=str(test_accuracy))


@captchami.route("/retrain/numbers", methods=['GET'])
def retrain_numbers():
    """
    Perform the training of the neural network so that it recognize the numbers in the captcha

    Returns:
        The accuracy on the test set
    """
    with open(config_file, "r") as conf:
        config = yaml.safe_load(conf)

    test_accuracy = train_numbers_net(config)
    return jsonify(result=str(test_accuracy))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    if args.debug:
        captchami.run(host=args.host, debug=True, port=args.port)
    else:
        captchami.run(host=args.host, debug=False, port=args.port)
