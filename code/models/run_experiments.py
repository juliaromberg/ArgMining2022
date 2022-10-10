import pandas as pd
import pickle
import os
from training import baseline, model_training

os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

# read in the dataset (including concreteness and subjectivity scores)
df = pd.read_csv("../../data/dataset+labels.csv")
# read in the prepared splits for the repeated 5-fold cross validation
splits = pickle.load(open("../../data/splits.pickle", "rb"))


def experiments4type(argument_type):
    """
    Run all(!) experiments for specified argument component type (except for BERT, see separate Notebook).
    The results will not be saved, just outputted in the terminal.

    :param argument_type: "joint", "mpos", or "premise"
    """
    print()
    print('Start experiments for predicting the concreteness...')
    print()

    baseline(df, argument_type, "concreteness")

    model_training(df, splits, argument_type, "LR", "length", "concreteness")
    model_training(df, splits, argument_type, "LR", "bow", "concreteness")
    model_training(df, splits, argument_type, "LR", "length+bow", "concreteness")

    model_training(df, splits, argument_type, "RF", "length", "concreteness")
    model_training(df, splits, argument_type, "RF", "bow", "concreteness")
    model_training(df, splits, argument_type, "RF", "length+bow", "concreteness")

    model_training(df, splits, argument_type, "SVM", "length", "concreteness")
    model_training(df, splits, argument_type, "SVM", "bow", "concreteness")
    model_training(df, splits, argument_type, "SVM", "length+bow", "concreteness")

    print()
    print('Start experiments for predicting the subjectivity (2 classes)...')
    print()

    baseline(df, argument_type, "subjectivity_2-class")

    model_training(df, splits, argument_type, "LR", "length", "subjectivity_2-class")
    model_training(df, splits, argument_type, "LR", "bow", "subjectivity_2-class")
    model_training(df, splits, argument_type, "LR", "length+bow", "subjectivity_2-class")

    model_training(df, splits, argument_type, "RF", "length", "subjectivity_2-class")
    model_training(df, splits, argument_type, "RF", "bow", "subjectivity_2-class")
    model_training(df, splits, argument_type, "RF", "length+bow", "subjectivity_2-class")

    model_training(df, splits, argument_type, "SVM", "length", "subjectivity_2-class")
    model_training(df, splits, argument_type, "SVM", "bow", "subjectivity_2-class")
    model_training(df, splits, argument_type, "SVM", "length+bow", "subjectivity_2-class")

    print()
    print('Start experiments for predicting the subjectivity (4 classes)...')
    print()

    baseline(df, argument_type, "subjectivity_4-class")

    model_training(df, splits, argument_type, "LR", "length", "subjectivity_4-class")
    model_training(df, splits, argument_type, "LR", "bow", "subjectivity_4-class")
    model_training(df, splits, argument_type, "LR", "length+bow", "subjectivity_4-class")

    model_training(df, splits, argument_type, "RF", "length", "subjectivity_4-class")
    model_training(df, splits, argument_type, "RF", "bow", "subjectivity_4-class")
    model_training(df, splits, argument_type, "RF", "length+bow", "subjectivity_4-class")

    model_training(df, splits, argument_type, "SVM", "length", "subjectivity_4-class")
    model_training(df, splits, argument_type, "SVM", "bow", "subjectivity_4-class")
    model_training(df, splits, argument_type, "SVM", "length+bow", "subjectivity_4-class")


if __name__ == '__main__':
    print('Joint analysis of the different argument component types (major position + premise).')
    experiments4type("joint")
    # print('Analysis of the argument component type major position.')
    # experiments4type("mpos")
    # print('Analysis of the argument component type premise.')
    # experiments4type("premise")
