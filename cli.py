import argparse


def parse_arguments(parser: argparse.ArgumentParser):
    main_parser = parser.add_subparsers(title="Run the Captchami service", description="Valid commands:",
                                        help="Sub command to train and load", dest="command")

    service = main_parser.add_parser("service", help="Create a service to run")
    service.add_argument("-port", type=int, default=6060,
                         help="Specify the port to use as service")
    service.add_argument("-debug", type=bool, default=False,
                         help="Define whether run the service with debug capabilities")
    service.add_argument("-host", type=str, default="0.0.0.0",
                         help="Define the host IP address")
    return parser.parse_args()
