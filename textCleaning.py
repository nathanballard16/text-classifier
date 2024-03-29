import pickle
import pandas as pd
import re
import sys
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
import numpy as np
import warnings
from sklearn.feature_selection import chi2

warnings.simplefilter(action='ignore', category=FutureWarning)

np.set_printoptions(threshold=sys.maxsize)


# load the dataset
# TODO: PLACE DATASET HERE **Must be a PICKLE FILE**

# path_df = "data/News_india_dataset.pickle"
#
# with open(path_df, 'rb') as data:
#     df = pickle.load(data)
#
# print(df.head())


# def create_train_test_pickles(df):
# fixing the data
# \r and \n
def clean_data(df, category_codes):
    df['Content_Parsed_1'] = df['text'].str.replace("\r", " ")
    df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace("\n", " ")
    df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace("    ", " ")

    # " when quoting text
    df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace('"', '')

    # Lowercasing the text
    df['Content_Parsed_2'] = df['Content_Parsed_1'].str.lower()

    punctuation_signs = list("?:!.,;")
    df['Content_Parsed_3'] = df['Content_Parsed_2']

    for punct_sign in punctuation_signs:
        df['Content_Parsed_3'] = df['Content_Parsed_3'].str.replace(punct_sign, '')

    df['Content_Parsed_4'] = df['Content_Parsed_3'].str.replace("'s", "")
    # print(df)

    stop_words = list(stopwords.words('english'))

    # Saving the lemmatizer into an object
    wordnet_lemmatizer = WordNetLemmatizer()

    nrows = len(df)
    lemmatized_text_list = []

    for row in range(0, nrows):

        # Create an empty list containing lemmatized words
        lemmatized_list = []

        # Get the list of all column names from headers
        # column_headers = list(df.columns.values)
        # print("The Column Header :", column_headers)

        # Save the text and its words into an object
        text = df['Content_Parsed_4'][row]
        # print(text)
        text_words = text.split(" ")

        # Iterate through every word to lemmatize
        for word in text_words:
            lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))

        # Join the list
        lemmatized_text = " ".join(lemmatized_list)

        # Append to the list containing the texts
        lemmatized_text_list.append(lemmatized_text)

    df['Content_Parsed_5'] = lemmatized_text_list

    df['Content_Parsed_6'] = df['Content_Parsed_5']

    for stop_word in stop_words:
        regex_stopword = r"\b" + stop_word + r"\b"
        df['Content_Parsed_6'] = df['Content_Parsed_6'].str.replace(regex_stopword, '')

    df = df.rename(columns={'file_name': 'File_Name', 'text': 'Content'})
    list_columns = ["File_Name", "Category", "Complete_Filename", "Content", "Content_Parsed_6"]
    df = df[list_columns]

    df = df.rename(columns={'Content_Parsed_6': 'Content_Parsed'})
    column_headers = list(df.columns.values)
    # print("The Column Header :", column_headers)
    # print(df)
    # category_codes = {
    #     'India_Assault': 0,
    #     'India_Fight': 1,
    #     'India_Protest': 2,
    #     'India_Threaten': 3,
    #     'India_UMV': 4
    # }
    #
    # Category mapping
    df['Category_Code'] = df['Category']
    df = df.replace({'Category_Code': category_codes})
    # # print(df.loc[[1313]])
    # # print(category_codes)
    # df['Category_Code'].to_csv('np.txt', sep='\t', index=False)
    for ind in df.index:
        if df['Category_Code'][ind] == 0:
            # print(df['Content'][ind])
            with open('data/GDELT_Labeled/RNN_Articles/RNN_1000/Assault/' + df['File_Name'][ind], 'w') as f:
                f.write(df['Content'][ind])
        elif df['Category_Code'][ind] == 1:
            # print(df['Content'][ind])
            with open('data/GDELT_Labeled/RNN_Articles/RNN_1000/Protest/' + df['File_Name'][ind], 'w') as f:
                f.write(df['Content'][ind])
        elif df['Category_Code'][ind] == 2:
            # print(df['Content'][ind])
            with open('data/GDELT_Labeled/RNN_Articles/RNN_1000/UMV/' + df['File_Name'][ind], 'w') as f:
                f.write(df['Content'][ind])
    X_train, X_test, y_train, y_test = train_test_split(df['Content_Parsed'],
                                                        df['Category_Code'],
                                                        test_size=0.15,
                                                        random_state=8)

    # Parameter election
    ngram_range = (1, 2)
    min_df = 10
    max_df = 1.
    max_features = 300

    tfidf = TfidfVectorizer(encoding='utf-8',
                            ngram_range=ngram_range,
                            stop_words=None,
                            lowercase=False,
                            max_df=max_df,
                            min_df=min_df,
                            max_features=max_features,
                            norm='l2',
                            sublinear_tf=True)

    features_train = tfidf.fit_transform(X_train).toarray()
    # y_train = y_train.astype('int')
    # print(y_train)
    labels_train = y_train
    # print(labels_train)
    # y_train = y_train.astype('int')
    # print(features_train.shape)

    features_test = tfidf.transform(X_test).toarray()
    labels_test = y_test
    # y_test = y_test.astype('int')
    # print(features_test.shape)

    for Product, category_id in sorted(category_codes.items()):
        features_chi2 = chi2(features_train, labels_train == category_id)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(tfidf.get_feature_names())[indices]
        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
        bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
        # print("# '{}' category:".format(Product))
        # print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-5:])))
        # print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-2:])))
        # print("")

    # X_train
    with open('data/GDELT_Labeled/train_test_1000/X_train.pickle', 'wb') as output:
        pickle.dump(X_train, output)

    # X_test
    with open('data/GDELT_Labeled/train_test_1000/X_test.pickle', 'wb') as output:
        pickle.dump(X_test, output)

    # y_train
    with open('data/GDELT_Labeled/train_test_1000/y_train.pickle', 'wb') as output:
        pickle.dump(y_train, output)

    # y_test
    with open('data/GDELT_Labeled/train_test_1000/y_test.pickle', 'wb') as output:
        pickle.dump(y_test, output)

    # df
    with open('data/GDELT_Labeled/train_test_1000/df.pickle', 'wb') as output:
        pickle.dump(df, output)

    # features_train
    with open('data/GDELT_Labeled/train_test_1000/features_train.pickle', 'wb') as output:
        pickle.dump(features_train, output)

    # labels_train
    with open('data/GDELT_Labeled/train_test_1000/labels_train.pickle', 'wb') as output:
        pickle.dump(labels_train, output)

    # features_test
    with open('data/GDELT_Labeled/train_test_1000/features_test.pickle', 'wb') as output:
        pickle.dump(features_test, output)

    # labels_test
    with open('data/GDELT_Labeled/train_test_1000/labels_test.pickle', 'wb') as output:
        pickle.dump(labels_test, output)

    # TF-IDF object
    with open('data/GDELT_Labeled/train_test_1000/tfidf.pickle', 'wb') as output:
        pickle.dump(tfidf, output)
