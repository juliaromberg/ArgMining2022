import pandas as pd
import numpy as np


def add_concreteness(df):
    concr_dict = {
        'high concreteness': 2,
        'intermediate concreteness': 1,
        'low concreteness': 0
    }
    inv_concr_dict = {v: k for k, v in concr_dict.items()}

    # map labels to numbers (0,1,2)
    df['concreteness'] = [[concr_dict[rating] for rating in e] for e in
                          df[['coder1', 'coder2', 'coder3', 'coder4', 'coder5']].values.tolist()]
    # mean of ratings and round again to three label values
    df['concreteness'] = [np.round(np.mean(e)) for e in df['concreteness']]
    # remap to class labels
    df['concreteness'] = df['concreteness'].apply(lambda x: inv_concr_dict[x])

    return df


def distance(ex):
    matrix = np.zeros((5, 5))

    for i in range(5):
        for j in range(5):
            matrix[i, j] = np.abs(ex[i] - ex[j]).sum()

    return int(np.sum(matrix))


def add_subjectivity(df):
    concr_dict = {
        'high concreteness': 2,
        'intermediate concreteness': 1,
        'low concreteness': 0
    }
    # define classes
    subj_dict_2class = {0: 0, 8: 0, 12: 1, 16: 1, 20: 1, 24: 1}
    subj_dict_4class = {0: 0, 8: 1, 12: 2, 16: 2, 20: 3, 24: 3}
    # map labels to numbers (0,1,2)
    df['concr_ratings'] = [[concr_dict[rating] for rating in e] for e in
                          df[['coder1', 'coder2', 'coder3', 'coder4', 'coder5']].values.tolist()]
    # calc distance value
    df['subjectivity_2-class'] = df['concr_ratings'].apply(lambda x: distance(x))
    df['subjectivity_4-class'] = df['concr_ratings'].apply(lambda x: distance(x))
    # map to six category labels
    df['subjectivity_2-class'] = df['subjectivity_2-class'].apply(lambda x: subj_dict_2class[x])
    df['subjectivity_4-class'] = df['subjectivity_4-class'].apply(lambda x: subj_dict_4class[x])

    counts = df_act['subjectivity'].value_counts()
    print(counts)

    return df


if __name__ == '__main__':

    datasets = ["CDB", "CDC", "CDM", "CQB", "MCK"]

    concreteness_list = []

    for d in datasets:
        '''
        The concreteness_"+d+".csv" files are not included here but can be found in
        the original dataset repository:
        https://github.com/juliaromberg/cimt-argument-concreteness-dataset
        '''
        concreteness = pd.read_csv("../../data/concreteness_" + d + ".csv")
        dataset = pd.read_csv("../../data/dataset_" + d + ".csv")

        text = []

        for index, row in concreteness.iterrows():
            document_id = row['document_id']
            sentences = row['sentences'].split('-')

            sent_ids = [i for i in range(int(sentences[0]), int(sentences[1]) + 1)]
            content = " ".join(dataset.loc[(dataset['document_id'] == document_id) &
                                           (dataset['sentence_nr'].isin(sent_ids))]['content'])

            text.append(content)

        concreteness['text'] = text
        concreteness_list.append(concreteness)

    df_act = pd.concat(concreteness_list, ignore_index=True)

    df_act = add_concreteness(df_act)
    df_act = add_subjectivity(df_act)

    # save labels
    df_act.to_csv("../../data/dataset+labels.csv", index=False)
