from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
import spacy
from spacy_iwnlp import spaCyIWNLP
from spacy.language import Language


def dummy(doc):
    return doc


class BOW(BaseEstimator, TransformerMixin):
    countVectorizer = None

    def __init__(self):
        self._lowercase = True
        self._stop_words = None
        self._ngram_range = (1, 1)
        self._analyzer = 'word'
        self._max_df = 1.0
        self._min_df = 1
        self._max_features = None
        pass

    def fit(self, X, y):
        # print("Learning new BOW structure ...")
        self.countVectorizer = CountVectorizer(tokenizer=dummy,
                                               preprocessor=dummy,
                                               lowercase=self._lowercase,
                                               stop_words=self._stop_words,
                                               ngram_range=self._ngram_range,
                                               analyzer=self._analyzer,
                                               max_df=self._max_df,
                                               min_df=self._min_df,
                                               max_features=self._max_features)
        self.countVectorizer.fit(X)

        return self

    def transform(self, X):
        if self.countVectorizer is None:
            raise Exception('no vocabulary learned yet. Run fit first')
        # print("Constructing BOW vectors ...")
        return self.countVectorizer.transform(X)


class Length(BaseEstimator, TransformerMixin):

    def __init__(self):
        global spacy_nlp
        if spacy_nlp is None:
            # print("Loading Spacy ...")
            spacy_nlp = spacy.load("de_core_news_sm")
        else:
            pass  # print("Spacy already loaded. Use loaded one.")
        pass

    def fit(self, X, y):
        return self

    def transform(self, X):
        global spacy_nlp
        if spacy_nlp is None:
            spacy_nlp = spacy.load("de_core_news_sm")
        # print("Start computing word tokens ...")
        return [[len([word.text for word in spacy_nlp(inst)])] for inst in X]


# globals for loading nlp tools only once at init and later reuse
spacy_nlp = None
spacy_iwnlp = None


@Language.component("iwnlp")
def my_component(doc):
    return doc


class SpacyTokens(BaseEstimator, TransformerMixin):

    def __init__(self):
        global spacy_nlp
        if spacy_nlp is None:
            # print("Loading Spacy ...")
            spacy_nlp = spacy.load("de_core_news_sm")
        else:
            pass  # print("Spacy already loaded. Use loaded one.")
        pass

    def fit(self, X, y):
        return self

    def transform(self, X):
        global spacy_nlp
        if spacy_nlp is None:
            spacy_nlp = spacy.load("de_core_news_sm")
        # print("Start computing word tokens ...")
        return [[word.text for word in spacy_nlp(inst)] for inst in X]


class SpacyTokensSW(BaseEstimator, TransformerMixin):

    def __init__(self):
        global spacy_nlp
        if spacy_nlp is None:
            # print("Loading Spacy ...")
            spacy_nlp = spacy.load("de_core_news_sm")
        else:
            pass  # print("Spacy already loaded. Use loaded one.")
        pass

    def fit(self, X, y):
        return self

    def transform(self, X):
        global spacy_nlp
        if spacy_nlp is None:
            spacy_nlp = spacy.load("de_core_news_sm")
        # print("Start computing word tokens ... Removing stop words ...")
        return [[word.text for word in spacy_nlp(inst) if not word.is_stop] for inst in X]


class SpacyTokensLemma(BaseEstimator, TransformerMixin):

    def __init__(self):
        global spacy_nlp
        if spacy_nlp is None:
            # print("Loading Spacy ...")
            spacy_nlp = spacy.load("de_core_news_sm")
        else:
            pass  # print("Spacy already loaded. Use loaded one.")

        global spacy_iwnlp
        if spacy_iwnlp is None:
            # print("Loading IWNLP ...")
            spacy_iwnlp = spaCyIWNLP(lemmatizer_path='/home/julia/Workspace/IWNLP.Lemmatizer_20181001.json')
            spacy_nlp.add_pipe("iwnlp")
        else:
            pass  # print("IWNLP already loaded. Use loaded one.")
        pass

    def fit(self, X, y):
        return self

    def transform(self, X):
        global spacy_nlp
        # print("Start computing word tokens ... Removing stop words ...")
        return [[word._.iwnlp_lemmas[0] if word._.iwnlp_lemmas else word.lemma_ for word in spacy_nlp(inst)] for inst in X]
