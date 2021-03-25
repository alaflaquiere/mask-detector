from argparse import ArgumentParser
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split

from data_utils.mask_dataset import MaskDataset
from utils.logger import Logger
from mask_detector import MaskClassifier

# TODO: plot the confusion matrix at the end of training


def train(dataFrame_p: Path, output_p: Path):

    # open pickle file
    with dataFrame_p.open("rb") as f:
        dataDf = pickle.load(f)

    # split into training and validation sets
    trainDf, valDf = train_test_split(
        dataDf, test_size=0.3, random_state=42, stratify=dataDf["mask"])
    train_dataset = MaskDataset(trainDf)
    val_dataset = MaskDataset(valDf)

    # create a logger
    logger = Logger(output_p / "logs")
    print("Follow progress on TensorBoard...")

    # create the model
    model = MaskClassifier(train_dataset, val_dataset,
                           save_path=output_p / "checkpoints",
                           logger=logger)

    # train
    print("Follow progress on Tensorboard")
    best_solution = model.fit()
    return best_solution


def clean_checkpoints(chkpt_p: Path, best_solution: dict):
    for i, file in enumerate(chkpt_p.glob("*.ckpt")):
        if i != best_solution["epoch"]:
            file.unlink()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-df",
                        default=Path("data", "RFMD_df.pkl"),
                        help="path the dataframe")
    parser.add_argument("-o",
                        default=Path("models", "trained_classifier"),
                        help="output directory")
    args = parser.parse_args()
    args.df = Path(args.df)
    args.o = Path(args.o)

    best_solution = train(args.df, args.o)

    clean_checkpoints(args.o / "checkpoints", best_solution)
