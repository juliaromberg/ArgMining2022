# ArgMining2022

About

This directory contains the code used for the publication "Is Your Perspective Also My Perspective? Enriching Prediction with Subjectivity", published in the Proceedings of the 9th Workshop on Argument Mining @ Coling 2022.

This work is based on research in the project CIMT/Partizipationsnutzen, which is funded by the Federal Ministry of Education and Research as part of its Social-Ecological Research funding priority, funding no. 01UU1904. (for more information, visit https://www.cimt-hhu.de/en/)

----------

Content

"code/data_preprocessing/" contains the code used to prepare the data basis by transforming the single concreteness ratings into a unified concreteness score and a subjectivity score (2 class schema and 4 class schema). Additionaly, the splits for the repeated 5-fold cross validation are prepared.

"code/data_analysis/" contains a script to get some statistics and correlation effects.

"code/models/" contains the code used for the experiments. "run_experiments" starts the experiments (baseline, logistic regression, random forest, SVM). "BERT.ipynb" is a Google Drive based Notebook to execute the BERT experiments.

"data/" contains the preprocessed dataset and splits. 

----------

Citation

@inproceedings{romberg-2022-perspective,
    title = "Is Your Perspective Also My Perspective? Enriching Prediction with Subjectivity",
    author = "Romberg, Julia",
    booktitle = "Proceedings of the 9th Workshop on Argument Mining",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "Association for Computational Linguistics",
    url = "?",
    pages = "?--?",
}


If you use the dataset, please cite the following paper:

@inproceedings{romberg-etal-2022-corpus,
    title = "A Corpus of {G}erman Citizen Contributions in Mobility Planning: Supporting Evaluation Through Multidimensional Classification",
    author = "Romberg, Julia and Mark, Laura and Escher, Tobias",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.308",
    pages = "2874--2883",
}

----------

License

The annotated data corpus is available under the Creative Commons CC BY-SA License (https://creativecommons.org/licenses/by-sa/4.0/).

----------

Contact Person

Julia Romberg, julia.romberg@hhu.de, https://www.cimt-hhu.de/en/team/romberg/
