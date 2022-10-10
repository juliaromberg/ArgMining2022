import pandas as pd
import numpy as np
import spacy
from scipy.stats import pearsonr, spearmanr


df = pd.read_csv("../../data/dataset+labels.csv")

print("Count units and split into argument component types:")
print("total, mpos, prem:", len(df), len(df[df['code'] == "mpos"]), len(df[df['code'] == "premise"]))

print(set(df['code']))

print()
print('++++sentences+++++')
lengths_total = [int(s[1]) - int(s[0]) + 1 for s in [i.split('-') for i in df['sentences']]]
lengths_mpos = [int(s[1]) - int(s[0]) + 1 for s in [i.split('-') for i in df[df['code'] == "mpos"]['sentences']]]
lengths_prem = [int(s[1]) - int(s[0]) + 1 for s in [i.split('-') for i in df[df['code'] == "premise"]['sentences']]]
print("average lengths of units (total) [mean, std, min, max]:", np.mean(lengths_total), np.std(lengths_total),
      np.min(lengths_total), np.max(lengths_total))  # nr sentences, nr tokens (avg+stddev, min, max)
print("average lengths of units (mpos) [mean, std, min, max]:", np.mean(lengths_mpos), np.std(lengths_mpos),
      np.min(lengths_mpos), np.max(lengths_mpos))  # nr sentences, nr tokens (avg+stddev, min, max)
print("average lengths of units (prem) [mean, std, min, max]:", np.mean(lengths_prem), np.std(lengths_prem),
      np.min(lengths_prem), np.max(lengths_prem))  # nr sentences, nr tokens (avg+stddev, min, max)

print()
print('++++tokens+++++')
nlp = spacy.load("de_core_news_sm")
lengths_total = [len([token for token in nlp(text)]) for text in df['text']]
df['lengths_total'] = lengths_total
lengths_mpos = [len([token for token in nlp(text)]) for text in df[df['code'] == "mpos"]['text']]
lengths_prem = [len([token for token in nlp(text)]) for text in df[df['code'] == "premise"]['text']]
print("average lengths of units (total) [mean, std, min, max]:", np.mean(lengths_total), np.std(lengths_total),
      np.min(lengths_total), np.max(lengths_total))  # nr sentences, nr tokens (avg+stddev, min, max)
print("average lengths of units (mpos) [mean, std, min, max]:", np.mean(lengths_mpos), np.std(lengths_mpos),
      np.min(lengths_mpos), np.max(lengths_mpos))  # nr sentences, nr tokens (avg+stddev, min, max)
print("average lengths of units (prem) [mean, std, min, max]:", np.mean(lengths_prem), np.std(lengths_prem),
      np.min(lengths_prem), np.max(lengths_prem))  # nr sentences, nr tokens (avg+stddev, min, max)


print()
print('++++coding combinations+++++')
df['codings_combined'] = df[['coder1', 'coder2', 'coder3', 'coder4', 'coder5']].values.tolist()
df['codings_combined'] = [sorted(le) for le in df['codings_combined']]

counts = df['codings_combined'].value_counts()
print(counts)


print()
print('++++label distribution+++++')
print(df['concreteness'].value_counts())
print()
print(df['subjectivity_2-class'].value_counts())
print()
print(df['subjectivity_4-class'].value_counts())


print()
print('++++correlation between concreteness score and subjectivity score++++')
concr_dict = {
    'high concreteness': 2,
    'intermediate concreteness': 1,
    'low concreteness': 0
}

print("2 class subjectivity")
df['concreteness_num'] = df['concreteness'].apply(lambda x: concr_dict[x])
corr, _ = pearsonr(df['concreteness_num'], df['subjectivity_2-class'])
print('Pearsons correlation: %.3f' % corr)
corr, _ = spearmanr(df['concreteness_num'], df['subjectivity_2-class'])
print('Spearmans correlation: %.3f' % corr)

print("4 class subjectivity")
corr, _ = pearsonr(df['concreteness_num'], df['subjectivity_4-class'])
print('Pearsons correlation: %.3f' % corr)
corr, _ = spearmanr(df['concreteness_num'], df['subjectivity_4-class'])
print('Spearmans correlation: %.3f' % corr)


print()
amc_dict = {
    'mpos': 0,
    'premise': 1,
}
df['type_num'] = df['code'].apply(lambda x: amc_dict[x])

print('++++correlation between argument type and concreteness score++++')
corr, _ = spearmanr(df['type_num'], df['concreteness_num'])
print('Spearmans correlation: %.3f' % corr)

print()
print('++++correlation between argument type and subjectivity score++++')
print("2 class subjectivity")
corr, _ = spearmanr(df['type_num'], df['subjectivity_2-class'])
print('Spearmans correlation: %.3f' % corr)

print("4 class subjectivity")
corr, _ = spearmanr(df['type_num'], df['subjectivity_4-class'])
print('Spearmans correlation: %.3f' % corr)


print()
print('++++correlation between text length and concreteness score++++')
corr, _ = spearmanr(df['lengths_total'], df['concreteness_num'])
print('Spearmans correlation: %.3f' % corr)

print()
print('++++correlation between text length and subjectivity score++++')
print("2 class subjectivity")
corr, _ = spearmanr(df['lengths_total'], df['subjectivity_2-class'])
print('Spearmans correlation: %.3f' % corr)

print("4 class subjectivity")
corr, _ = spearmanr(df['lengths_total'], df['subjectivity_4-class'])
print('Spearmans correlation: %.3f' % corr)