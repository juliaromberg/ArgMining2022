import numpy as np


def avg_results(results, labels):
    f1_scores = {label: np.mean([d[str(label)]['f1-score'] for d in results]) for label in labels}
    f1_std = {label: np.std([d[str(label)]['f1-score'] for d in results]) for label in labels}

    acc = np.mean([d['accuracy'] for d in results])
    acc_std = np.std([d['accuracy'] for d in results])
    macro = np.mean([d['macro avg']['f1-score'] for d in results])
    macro_std = np.std([d['macro avg']['f1-score'] for d in results])

    print("f1:", [str(l) + ": " + str(round(f1_scores[l], 2)) + " +- " + str(round(f1_std[l], 2)) for l in labels])
    print("acc:", round(acc, 2), round(acc_std, 2))
    print("macro f1:", round(macro, 2), round(macro_std, 2))