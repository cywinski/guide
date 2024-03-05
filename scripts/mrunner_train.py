from argparse import Namespace

from mrunner.helpers.client_helper import get_configuration

from scripts.image_train import run_training_with_args


def main():
    params = get_configuration(print_diagnostics=True, with_neptune=False)
    run_training_with_args(Namespace(**params))


if __name__ == "__main__":
    main()
