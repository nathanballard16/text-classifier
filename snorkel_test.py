'''
https://medium.com/data-science-in-your-pocket/programmatically-labeling-data-using-snorkel-with-example-a6a322ef0f2c
'''
from gibberish import Gibberish
import enchant
import random
import string
import pandas as pd
import numpy as np
import itertools
import re
from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model.label_model import LabelModel

gib = Gibberish()
eng_dict = enchant.Dict("en_US")


def generate_data(count):
    gib_words = gib.generate_words(count)
    eng_words = list(itertools.chain(
        *[eng_dict.suggest("".join([random.choice(string.ascii_lowercase) for _ in range(random.randint(3, 10))])) for _
          in range(0, count)]))

    words, labels = [x.lower() for x in gib_words + eng_words], list(np.zeros(len(gib_words), dtype=np.int8)) + list(
        np.ones(len(eng_words), dtype=np.int8))

    return pd.DataFrame(data={'word': words}), np.array(labels)


train_df, _ = generate_data(1000)
validate_df, validate_labels = generate_data(1000)

ABSTAIN = -1
DICT_WORD = 1
GIBBERISH = 0

print(train_df)


@labeling_function()
def no_vowel(record):
    if sum([1 if x in record['word'] else 0 for x in ['a', 'e', 'i', 'o', 'u']]) == 0:
        return GIBBERISH
    else:
        return ABSTAIN


@labeling_function()
def not_all_vowels(record):
    if sum([1 if x in ['a', 'e', 'i', 'o', 'u'] else 0 for x in record['word']]) < len(record['word']):
        return DICT_WORD
    else:
        return ABSTAIN


@labeling_function()
def length(record):
    if len(record['word']) < 8:
        return DICT_WORD
    else:
        return GIBBERISH


@labeling_function()
def consecutive_consonants(record):
    if re.findall(r'(?:(?![aeiou])[a-z]){3,}', record['word']):
        return GIBBERISH
    else:
        return DICT_WORD


@labeling_function()
def consecutive_vowels(record):
    if re.findall(r"\b(?=[a-z]*[aeiou]{3,})[a-z]+\b", record['word']):
        return GIBBERISH
    else:
        return DICT_WORD


lfs = [no_vowel, not_all_vowels, length, consecutive_consonants, consecutive_vowels]
applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=train_df)
L_validate = applier.apply(df=validate_df)

print(L_train)

LFAnalysis(L_train, lfs).lf_summary()

label_model = LabelModel(verbose=False)
label_model.fit(L_train=L_train, n_epochs=1000, seed=100)
preds_train_label = label_model.predict(L=L_train)
preds_valid_label = label_model.predict(L=L_validate)

print('validate metrics')
print(label_model.score(L_validate, Y=validate_labels, metrics=["f1", "accuracy", 'precision', 'recall']))

print(LFAnalysis(L_validate, lfs).lf_summary(validate_labels))
