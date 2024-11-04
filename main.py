import argparse
import sys
import cv2

from update_dataset import update_dataset
from clear_dataset_feature import clear_dataset_feature
from functions.classifiers.bayes.classifier import BAYES_classify
from functions.classifiers.bayes.check_correlation import BAYES_check_correlation
from functions.classifiers.result import print_classification_result
from functions.features import ImageFeatures


def args_def() -> tuple[argparse.ArgumentParser, dict[str, argparse.ArgumentParser]]:
    args = argparse.ArgumentParser(prog="leaf_classifier")

    subparsers = args.add_subparsers(
        dest="command",
    )

    subparsers.add_parser(
        name="update",
        help="update the dataset after adding some images or features",
    )

    remove_feature = subparsers.add_parser(
        name="rmfeature",
        help="remove a feature from the cached values",
    )
    remove_feature.add_argument(
        "--feature",
        "-f",
        type=str,
        action="store",
        help="the name of the feature to be removed (mandatory)",
    )

    classify = subparsers.add_parser(
        name="classify",
        help="perform the classification task on a picture",
    )
    classify.add_argument(
        "--img",
        "-i",
        type=str,
        action="store",
        help="the path to the image to be classified (mandatory)",
    )

    subparsers.add_parser(
        name="correlation",
        help="show the correlation matrix for all the features",
    )

    return (args, {"rm": remove_feature, "c": classify})


if __name__ == "__main__":
    args_parser, subparsers = args_def()
    args = args_parser.parse_args(sys.argv[1:])

    if args.command == "update":
        update_dataset()

    elif args.command == "rmfeature":
        if args.feature == None:
            subparsers["rm"].print_help()
        else:
            clear_dataset_feature(args.feature)

    elif args.command == "classify":
        if args.img == None:
            subparsers["c"].print_help()
        else:
            print("Starting analizing picture...")
            img = ImageFeatures(args.img)
            print_classification_result(BAYES_classify(img.get_features()))

    elif args.command == "correlation":
        BAYES_check_correlation()

    else:
        args_parser.print_help()
