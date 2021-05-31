import torch
import random
import os
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

from captchami.loaders import CaptchaDataset
from captchami.neural_net import NeuralNet
from captchami.vision import *

captcha_service = Flask(__name__)
cors = CORS(captcha_service, resources={r"/*": {"origins": "*"}})
mapper = {0: "+", 1: "-", 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7}
os.chdir("/home/fabio/CaptchAmI")
bin_net = Path("./bin_net.pt")
number_net = Path("./numbers_net.pt")
temp_file = Path("./temp.png")


@captcha_service.route("/classify/", methods=['POST'])
def classify():
    """
    This function takes as input a JSON file with a field called "base64_img" and elaborates it in order to find if
    there is an operation or a bunch of stars.
    It loads the datasets containing the stars and the number to get the right sizes of the image and perform two
    different classification: one to determine whether the image contains stars or not (binary classification) and then
    it chooses the correct neural network to use to classify the file.

    Returns: a number which is either the result of the operation or the sum of all the stars
    """

    bin_loader = CaptchaDataset(Path("./datasets/binary"))
    num_loader = CaptchaDataset(Path("./datasets/multiclass/numbers"))
    content = request.json
    base64_img = content["base64_img"]
    base64_to_img(base64_img, temp_file)
    nn = NeuralNet(l_i=6400, classes=bin_loader.get_classes(), loaders=bin_loader)
    nn.load(bin_net)
    classed = nn.classify_file(temp_file)

    if classed == 0:
        captcha_service.logger.info("Received a number")
        # classed == 0 means that we have a number to elaborate
        nn = NeuralNet(l_i=720, classes=num_loader.get_classes(), loaders=num_loader)
        nn.load(number_net)
        try:
            elements = elaborate_numbers(temp_file)
        except ValueError:
            captcha_service.logger.error("Value error on matching elements array. Guessing ...")
            result = random.randint(1, 8)
            return jsonify(result=str(result))

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
                # store_misclassified(temp_file)
                captcha_service.logger.error("Type error occurred on + operator, guessing...")
                result = random.randint(1, 8)
        else:
            try:
                result = e1 - e2
            except TypeError:
                # store_misclassified(temp_file)
                captcha_service.logger.error("Type error occurred on - operator, guessing...")
                result = random.randint(1, 8)
    else:
        captcha_service.logger.info("Received some stars")
        # Use CV to classify and get the numbers
        result = elaborate_stars(temp_file)

    if int(result) <= 0:
        captcha_service.logger.error("<= 0 error, guessing...")
        # store_misclassified(temp_file)
        result = random.randint(1, 8)
    captcha_service.logger.info("New classification: " + str(result))

    return jsonify(result=str(result))
