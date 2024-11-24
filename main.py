import argparse
import sys
import os

from update_dataset import update_dataset
from clear_dataset_feature import clear_dataset_feature
from functions.classifiers.bayes.classifier import BAYES_classify
from functions.classifiers.bayes.check_correlation import (
    BAYES_check_correlation,
    BAYES_check_ABS_correlation,
)
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
        help="the name of the model feature to be removed",
    )
    remove_feature.add_argument(
        "--internal",
        "-i",
        type=str,
        action="store",
        help="the name of the internal feature to be removed",
        metavar="FEATURE",
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
        help="the path to the image to be classified",
    )
    classify.add_argument(
        "--dir",
        "-d",
        type=str,
        action="store",
        help="the path to a folder whose content should be classified (non-recursive)",
    )

    correlation = subparsers.add_parser(
        name="correlation",
        help="show the correlation matrix for all the features",
    )
    correlation.add_argument(
        "--abs",
        action="store_true",
        help="Display the absolute value of the correlation, instead of the positive and negative correlation",
    )

    return (args, {"rm": remove_feature, "c": classify})


if __name__ == "__main__":
    args_parser, subparsers = args_def()
    args = args_parser.parse_args(sys.argv[1:])

    if args.command == "update":
        update_dataset()

    elif args.command == "rmfeature":
        if args.feature == None and args.internal == None:
            subparsers["rm"].print_help()
        else:
            clear_dataset_feature(args.feature, "features")
            clear_dataset_feature(args.internal, "internal")

    elif args.command == "classify":
        if args.img == None and args.dir == None:
            subparsers["c"].print_help()
        elif args.img != None:
            print("Starting analizing picture...")
            img = ImageFeatures(args.img)
            print_classification_result(BAYES_classify(img.get_features()))
        else:
            for img_name in os.listdir(args.dir):
                print(f'Starting analizing picture "{img_name}"...')
                try:
                    img = ImageFeatures(f"{args.dir}/{img_name}")
                    print_classification_result(BAYES_classify(img.get_features()))
                except AttributeError:
                    print(f'"{img_name}" is not an image')
                print("============================================================")

    elif args.command == "correlation":
        if args.abs:
            BAYES_check_ABS_correlation()
        else:
            BAYES_check_correlation()

    else:
        args_parser.print_help()
