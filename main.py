import torch
from waitress import serve

from captchami.loaders import CaptchaDataset
from captchami.neural_net import NeuralNet
from captchami.vision import *
from cli import *
from restapi.service import captcha_service


def main():
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    if args.command == "train":
        loaders = CaptchaDataset(args.dataset)
        if "binary" in args.dataset:
            l_i = 6400
        else:
            l_i = 720

        nn = NeuralNet(l_i=l_i, classes=loaders.get_classes(), loaders=loaders)
        # We start the training of the neural network and save the models
        nn.train()
        nn.save(path=args.o)

    elif args.command == "classify":
        # First check if it is a number or stars
        loaders = CaptchaDataset(args.b_dataset)
        nn = NeuralNet(l_i=6400, classes=loaders.get_classes(), loaders=loaders)
        nn.load(args.b_nn)
        classed = nn.classify_file(args.file)

        if classed == 0:
            # We have a number to classify
            loaders = CaptchaDataset(args.n_dataset)
            nn = NeuralNet(l_i=720, classes=loaders.get_classes(), loaders=loaders)
            nn.load(args.n_nn)
            elements = elaborate_numbers(args.file)
            parsed = []
            for e in elements:
                e = np.asarray(e[1]).astype(int)*255
                e = torch.Tensor(e)
                result = nn.classify_img(e)
                parsed.append(str(result))
            print("Elements found are: ", parsed)

        else:
            # Use CV to classify and get the numbers
            result = elaborate_stars(args.file)
            print("File classified as: ", result)

    elif args.command == "service":
        if args.debug:
            captcha_service.run(host='0.0.0.0', debug=True, port=args.port)
        else:
            serve(captcha_service, port=args.port)


if __name__ == "__main__":
    main()
