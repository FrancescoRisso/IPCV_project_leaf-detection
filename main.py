from update_dataset import update_dataset
from functions.classifiers.bayes.summarize_dataset import BAYES_summarize_dataset
from functions.classifiers.bayes.check_correlation import BAYES_check_correlation
from functions.classifiers.bayes.classifier import BAYES_classify
from functions.classifiers.result import print_classification_result
from functions.features import ImageFeatures
import cv2

if __name__ == "__main__":
    # update_dataset()
    # BAYES_summarize_dataset()
    # BAYES_check_correlation()

    plant = "gaggia-data"
    print(f'Programmer\'s knowledge: plant is "{plant}"\n')
    f = ImageFeatures(cv2.imread(f"./test/{plant}.jpg"))
    print_classification_result(BAYES_classify(f.get_features()))
