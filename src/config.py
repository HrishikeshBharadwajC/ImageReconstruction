import argparse

parser = argparse.ArgumentParser(description="final_project")
parser.add_argument("--dataset-path", type=str, help="path to dataset folder")
parser.add_argument(
    "--batch-size",
    type=int,
    default=64,
    help="input batch size for training (default: 64)",
)
parser.add_argument(
    "--epochs", type=int, default=100, help="number of epochs to train (default: 100)"
)
parser.add_argument(
    "--learning-rate-discriminator",
    type=float,
    default=0.0002,
    help="Learning rate of discriminator",
)
parser.add_argument(
    "--learning-rate-generator",
    type=float,
    default=0.0002,
    help="Learning rate of generator",
)
parser.add_argument(
    "--lamda",
    type=int,
    default=100,
    help="Multiplication factor of L1 loss in loss computation of generator",
)
parser.add_argument("--num-workers", type=int, default=2, help="Number of workers")
parser.add_argument(
    "--save-every-n-epochs",
    type=int,
    default=5,
    help="Every how many epochs should the results be saved",
)
parser.add_argument(
    "--load-model-from-epoch",
    type=int,
    default=0,
    help="Use when loading from checkpoint. Enter a epoch number from which we need to load checkpoint",
)
parser.add_argument(
    "--dataset-direction", 
    type=str, default="normal", 
    help="The direction of image converstion in the dataset. The default value is normal meaning the first image will be the input and second will be the target. For the other way round the direction should be set to reverse."
)
hyperparams = parser.parse_args()
hyperparameters = vars(hyperparams)
