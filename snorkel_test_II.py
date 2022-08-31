'''
https://www.section.io/engineering-education/snorkel-python-for-labeling-datasets-programmatically/
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from snorkel.labeling import labeling_function
import re
from snorkel.labeling import PandasLFApplier
from snorkel.labeling.model import LabelModel

data = pd.read_table('./data/unlabelled-dataset.txt')
data = data.sample(frac=1).reset_index(drop=True)
data.columns = ["sentences"]

print(data.head())

df_train, df_test = train_test_split(data, train_size=0.5)
print(df_train.shape)

QUESTION = 1
ABSTAIN = -1


@labeling_function()
def lf_keyword_lookup(x):
    keywords = ["why", "what", "when", "who", "where", "how"]
    return QUESTION if any(word in x.sentences.lower() for word in keywords) else ABSTAIN


@labeling_function()
def lf_regex_contains_what(x):
    return QUESTION if re.search(r"what.*?", x.sentences, flags=re.I) else ABSTAIN


@labeling_function()
def lf_regex_contains_question_mark(x):
    return QUESTION if re.search(r".*?", x.sentences, flags=re.I) else ABSTAIN


lfs = [lf_keyword_lookup, lf_regex_contains_what, lf_regex_contains_question_mark]

applier = PandasLFApplier(lfs=lfs)

L_train = applier.apply(df=df_train)

label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)

df_train['Labels'] = label_model.predict(L=L_train, tie_break_policy="abstain")

print(df_train)
