import os
import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from snorkel.labeling import LFAnalysis
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import labeling_function
from snorkel.labeling.model.label_model import LabelModel
from snorkel.labeling.model import MajorityLabelVoter
from textblob import TextBlob
import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
import spacy
import locationtagger
from tqdm import tqdm
import geocoder

# essential entity models downloads
nltk.downloader.download('maxent_ne_chunker')
nltk.downloader.download('words')
nltk.downloader.download('treebank')
nltk.downloader.download('maxent_treebank_pos_tagger')
nltk.downloader.download('punkt')
nltk.download('averaged_perceptron_tagger')

ABSTAIN = -1
NOT_UNREST = 0
UNREST = 1
global_df = pd.DataFrame()
active_lfs = []


def read_config():
    # Reading in the YAML config file
    print("Reading Config File ...")
    with open("Snorkel_config/Snorkel_Config.yaml", "r") as yamlfile:
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)
        print("Read Config File Successful")
    return data


def create_lfs(data):
    print('Creating LFS from active methods ...')
    if data['Rules'][0]['Dataset_Location']['status']: active_lfs.append(index1)
    if data['Rules'][1]['Article_Length']['status']: active_lfs.append(index2)
    if data['Rules'][2]['GDELT_Definition']['status']: active_lfs.append(index3)
    if data['Rules'][3]['Common_Words']['status']: active_lfs.append(index4)
    if data['Rules'][4]['Named_Entities']['status']: active_lfs.append(index5)
    if data['Rules'][5]['Sentiment']['status']: active_lfs.append(index6)


def read_labeled_data(data):
    df = pd.read_csv(data['Data'][1]['Labeled']['file'])

    return df.sample(n=int(data['Data'][0]['Raw_Data']['total_files']))


def read_unlabeled_data(data):
    files_folder = str(data['Data'][0]['Raw_Data']['path'])
    file_names = os.listdir(files_folder)
    # Create Dictionary for File Name and Text
    file_name_and_text = {}
    for file in file_names:
        with open(files_folder + '/' + file, "r") as target_file:
            file_name_and_text[file] = target_file.read()
    file_data = (pd.DataFrame.from_dict(file_name_and_text, orient='index')
                 .reset_index().rename(index=str, columns={'index': 'file_name', 0: 'text'}))
    file_data['text_length'] = file_data['text'].str.len()
    file_data['max_length'] = int(file_data['text_length'].quantile(.95))

    return file_data.sample(n=int(data['Data'][0]['Raw_Data']['total_files']))


def average_sentiment(data):
    all_sentiments = []
    for ind in tqdm(data.index, desc="Calculating Average Sentiment"):
        sentiment = TextBlob(str(data['text'][ind])).sentiment.polarity
        all_sentiments.append(sentiment)
    return sum(all_sentiments) / len(all_sentiments)


def get_locations(data):
    total_locations = []
    for ind in tqdm(data.index, desc="Calculating Locations in text"):
        place_entity = locationtagger.find_locations(text=data['text'][ind])
        total_locations.append(place_entity.countries)
    return total_locations


def get_named_entities(data):
    orgs = []
    for ind in tqdm(data.index, desc="Calculating Named Entities in text"):
        nltk_results = ne_chunk(pos_tag(word_tokenize(data['text'][ind])))
        for nltk_result in nltk_results:
            if type(nltk_result) == Tree:
                name = ''
                for nltk_result_leaf in nltk_result.leaves():
                    name += nltk_result_leaf[0] + ' '
                    if nltk_result.label() == "ORGANIZATION":
                        orgs.append(name)
    count = pd.Series(orgs).value_counts()
    print(count[0:10])


# def merge_data_sources(first_df):
#     files_folder = 'D:/Graduate School/text-classifier/data/GoldenDataset/Assault'
#     file_names = os.listdir(files_folder)
#     # Create Dictionary for File Name and Text
#     file_name_and_text = {}
#     for file in file_names:
#         with open(files_folder + '/' + file, "r") as target_file:
#             file_name_and_text[file] = target_file.read()
#     file_data = (pd.DataFrame.from_dict(file_name_and_text, orient='index')
#                  .reset_index().rename(index=str, columns={'index': 'file_name', 0: 'text'}))
#     file_data['text_length'] = file_data['text'].str.len()
#     # file_data['max_length'] = int(file_data['text_length'].quantile(.95))
#     df_all = file_data.merge(first_df.drop_duplicates(), on=['file_name', 'text'],
#                              how='left', indicator=True)
#     df1_only = df_all[df_all['_merge'] == 'left_only']
#     df1_only = df1_only.drop('_merge', axis=1)
#     df1_only = df1_only.drop('text_length_y', axis=1)
#     df1_only = df1_only.rename(columns={"text_length_x": "text_length"})
#
#     final_df = pd.concat([first_df, df1_only])
#     final_df['labels'] = np.where(final_df['labels'] != 1, int(0), final_df['labels'])
#     final_df['max_length'] = int(file_data['text_length'].quantile(.95))
#     final_df.to_csv('./data/GDELT_Labeled/assault_labeled.csv', sep=',')


@labeling_function()
def dataset_location(x):
    keywords = ["India", "Bangladesh", "Pakistan"]
    total = 0
    for cities in x.locations:
        if cities in keywords:
            total += 1
    return UNREST if total > 0 else ABSTAIN


@labeling_function()
def article_length(x):
    return UNREST if x.text_length <= x.max_length else ABSTAIN


@labeling_function()
def gdelt_definition(x):
    pass


@labeling_function()
def common_words(x):
    pass


@labeling_function()
def named_entities(x):
    pass


@labeling_function()
def sentiment_score(x):
    return UNREST if TextBlob(x.text).sentiment.polarity < (x.average_sentiment + 0.25) else ABSTAIN


def label_data(labeled_df, unlabeled_df):
    # print(df)
    lfs = active_lfs
    df_train = unlabeled_df
    validate_df = labeled_df
    validate_labels = labeled_df['labels'].to_numpy()
    # Apply the LFs to the unlabeled training data
    applier = PandasLFApplier(lfs)
    L_train = applier.apply(df_train)
    L_validate = applier.apply(df=validate_df)
    print(LFAnalysis(L_train, lfs).lf_summary())
    # Train the label model and compute the training labels
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train, n_epochs=500, log_freq=50, seed=123)
    df_train["label"] = label_model.predict(L=L_train, tie_break_policy="abstain")
    # print('The Entire Dataset is Labeled 1: ' + str((df_train["label"] == 1).all()))
    # print(df_train)
    # print(df_train[df_train["label"] != 1])
    print('validate metrics')
    print(label_model.score(L_validate, Y=validate_labels, metrics=["f1", "accuracy", 'precision', 'recall']))
    print(LFAnalysis(L_validate, lfs).lf_summary(validate_labels))


index1, index2, index3, index4, index5, index6 = eval('dataset_location'), eval('article_length'), eval(
    'gdelt_definition'), \
    eval('common_words'), eval('named_entities'), eval('sentiment_score')


def main():
    data = read_config()
    create_lfs(data)
    labeled_df = read_labeled_data(data)
    raw_data_df = read_unlabeled_data(data)
    print()
    print('Getting Average Sentiment ...')
    sentiment = average_sentiment(raw_data_df)
    raw_data_df['average_sentiment'] = sentiment
    labeled_df['average_sentiment'] = sentiment
    print()
    print('Getting Locations in Text ...')
    locations = get_locations(raw_data_df)
    labeled_locations = get_locations(labeled_df)
    raw_data_df['locations'] = locations
    labeled_df['locations'] = labeled_locations
    print(raw_data_df)
    print()
    print('Getting Named Entities ... ')
    get_named_entities(raw_data_df)
    label_data(labeled_df, raw_data_df)


if __name__ == "__main__":
    main()
