import argparse


def parse_arguments(parser: argparse.ArgumentParser):
    main_parser = parser.add_subparsers(title="Train and load the neural network", description="Valid commands:",
                                        help="Sub command to train and load", dest="command")

    # Train the neural network
    train = main_parser.add_parser("train", help="Train the neural network")
    train.add_argument("-dataset", type=str, default="./datasets/binary",
                       help="Specify the root folder for the dataset")
    train.add_argument("-o", type=str, default="./b_net.pt",
                       help="Specify the file to save the neural network")

    # Use the neural network
    classify = main_parser.add_parser("classify", help="Load a trained model")
    classify.add_argument("-b_dataset", type=str, default="./datasets/binary",
                          help="Specify the root folder for the BINARY dataset")
    classify.add_argument("-n_dataset", type=str, default="./datasets/multiclass/numbers",
                          help="Specify the root folder for the NUMBERS dataset")
    classify.add_argument("-b_nn", type=str, default="./bin_net.pt",
                          help="Specify the .pkl file containing the BINARY trained neural network")
    classify.add_argument("-n_nn", type=str, default="./numbers_net.pt",
                          help="Specify the .pkl file containing the NUMBER trained neural network")
    classify.add_argument("-file", type=str, default="./misclassified/2.png",
                          help="The file to classify")

    service = main_parser.add_parser("service", help="Create a service to run")
    service.add_argument("-port", type=int, default=6060,
                         help="Specify the port to use as service")
    service.add_argument("-debug", type=bool, default=False,
                         help="Define whether run the service with debug capabilities")
    return parser.parse_args()
