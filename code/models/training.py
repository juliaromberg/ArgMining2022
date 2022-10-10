import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import PredefinedSplit
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from features import *
from evaluation import avg_results


classifier_choices = {
    "LR": LogisticRegression(),
    "RF": RandomForestClassifier(random_state=1, n_jobs=-1),
    "SVM": SVC()
}

param_grid_choices = {
    "LR": [
            {'classifier__penalty': ['l1', 'l2'],
             'classifier__C': [0.001, 0.1, 1, 10, 100],
             'classifier__solver': ['liblinear'],
             'classifier__class_weight': ['balanced', None]}
        ],
    "RF": [
            {'classifier__criterion': ['gini', 'log_loss'],
             'classifier__class_weight': ['balanced', None, 'balanced_subsample']}
        ],
    "SVM": [
            {'classifier__kernel': ['rbf'],
             'classifier__class_weight': ['balanced'],
             'classifier__C': [0.1, 1, 10, 100],
             'classifier__gamma': [1, 0.1, 0.01, 0.001]}
        ]
}

feature_choices = {
    "length": ('length', Pipeline([
        ('text_length', Length())
    ])),
    "bow": ('unigram', Pipeline([
        ('spacy_tokenizer', SpacyTokens()),
        ('bow', BOW())
    ])),
    "length+bow": ('feature_union', FeatureUnion([
        ('unigram', Pipeline([
            ('spacy_tokenizer', SpacyTokens()),
            ('bow', BOW())
        ])),
        ('length', Pipeline([
            ('text_length', Length())
        ])),
    ]))
}


def baseline(df, argument_type, target):
    print("Baseline (Majority Vote)")

    if argument_type in ["mpos", "premise"]:
        df_target = df[df['code'] == argument_type].reset_index(drop=True)
        y = df_target[target]
    else:
        y = df[target]

    if target == "concreteness":
        concr_dict = {
            'high concreteness': 2,
            'intermediate concreteness': 1,
            'low concreteness': 0
        }
        y = y.apply(lambda x: concr_dict[x])
        print(classification_report(y, [2] * len(y)))
    else:
        print(classification_report(y, [0] * len(y)))


def model_training(df, pred_splits, argument_type, classifier, features, target):
    """
    :param df: data
    :param pred_splits: pre-defined splits for the repeated cross-validation
    :param argument_type: argument component type ("joint", "mpos", or "premise")
    :param classifier: "LR" (logistic regression), "RF" (random forest), or "SVM"
    :param features: "length", "bow", or "length+bow"
    :param target: "concreteness", "subjectivity_2-class", or "subjectivity_4-class"
    """

    print(classifier, features)

    # define algorithm, parameters for grid search, and the classification pipeline
    cl_algorithm = classifier_choices[classifier]
    param_grid = param_grid_choices[classifier]
    pipe = Pipeline([feature_choices[features], ('classifier', cl_algorithm)])

    # prepare data and splits
    X, y = df['text'], df[target]
    splits = pred_splits[argument_type]
    split_num = 1

    results_overall = []

    for k, v in splits.items():
        print("Split", split_num)
        # split the dataframe indices according to train, val and test
        train_index, val_index, test_index = v['train_index'], v['val_index'], v['test_index']

        # join train and val set into dataframe to assign continuous indices
        df_train = pd.DataFrame.from_dict(
            {'index': train_index, 'X': X[train_index], 'y': y[train_index].tolist(), 'test_fold': -1})
        df_val = pd.DataFrame.from_dict(
            {'index': val_index, 'X': X[val_index], 'y': y[val_index].tolist(), 'test_fold': 0})
        df_train_val = pd.concat([df_train, df_val], ignore_index=True)

        # get features and labels for train and test
        X_train_val, y_train_val = df_train_val['X'], df_train_val['y']
        X_test, y_test = X[test_index], y[test_index]

        # set predefined validation set
        test_fold = df_train_val['test_fold'].to_numpy()
        ps = PredefinedSplit(test_fold=test_fold)

        # train and evaluate classifier
        clf = GridSearchCV(pipe, param_grid=param_grid, cv=ps, verbose=3, n_jobs=-1)
        model = clf.fit(X_train_val, y_train_val)
        y_pred = model.predict(X_test)
        results_overall.append(classification_report(y_test, y_pred, output_dict=True))

        split_num += 1

    print("AVG Overall:")
    avg_results(results_overall, list(set(y)))
