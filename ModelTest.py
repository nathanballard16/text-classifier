import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import punkt
from nltk.corpus.reader import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

path_models = "models/"

# SVM
path_svm = path_models + 'best_svc.pickle'
path_gbc = path_models + 'best_gbc.pickle'
path_knnc = path_models + 'best_knnc.pickle'
path_lrc = path_models + 'best_lrc.pickle'
path_mnbc = path_models + 'best_mnbc.pickle'
path_rfc = path_models + 'best_rfc.pickle'

with open(path_svm, 'rb') as data:
    svc_model = pickle.load(data)
with open(path_gbc, 'rb') as data:
    gbc_model = pickle.load(data)
with open(path_knnc, 'rb') as data:
    knnc_model = pickle.load(data)
with open(path_lrc, 'rb') as data:
    lrc_model = pickle.load(data)
with open(path_mnbc, 'rb') as data:
    mnbc_model = pickle.load(data)
with open(path_rfc, 'rb') as data:
    rfc_model = pickle.load(data)

path_tfidf = "train_test/tfidf.pickle"
with open(path_tfidf, 'rb') as data:
    tfidf = pickle.load(data)

category_codes = {
    'business': 0,
    'entertainment': 1,
    'politics': 2,
    'sport': 3,
    'tech': 4
}

punctuation_signs = list("?:!.,;")
stop_words = list(stopwords.words('english'))


def create_features_from_text(text):
    # Dataframe creation
    lemmatized_text_list = []
    df = pd.DataFrame(columns=['Content'])
    df.loc[0] = text
    df['Content_Parsed_1'] = df['Content'].str.replace("\r", " ")
    df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace("\n", " ")
    df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace("    ", " ")
    df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace('"', '')
    df['Content_Parsed_2'] = df['Content_Parsed_1'].str.lower()
    df['Content_Parsed_3'] = df['Content_Parsed_2']
    for punct_sign in punctuation_signs:
        df['Content_Parsed_3'] = df['Content_Parsed_3'].str.replace(punct_sign, '')
    df['Content_Parsed_4'] = df['Content_Parsed_3'].str.replace("'s", "")
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized_list = []
    text = df.loc[0]['Content_Parsed_4']
    text_words = text.split(" ")
    for word in text_words:
        lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
    lemmatized_text = " ".join(lemmatized_list)
    lemmatized_text_list.append(lemmatized_text)
    df['Content_Parsed_5'] = lemmatized_text_list
    df['Content_Parsed_6'] = df['Content_Parsed_5']
    for stop_word in stop_words:
        regex_stopword = r"\b" + stop_word + r"\b"
        df['Content_Parsed_6'] = df['Content_Parsed_6'].str.replace(regex_stopword, '')
    df = df['Content_Parsed_6']
    # df.to_csv('np.txt', sep='\t', index=False)
    # df = df.rename(columns={'Content_Parsed_6': 'Content_Parsed'})
    # df.rename(index=str, columns={'Content_Parsed_6': 'Content_Parsed'})

    # TF-IDF
    features = tfidf.transform(df).toarray()

    return features


def get_category_name(category_id):
    for category, id_ in category_codes.items():
        if id_ == category_id:
            return category


def predict_from_text_svc(text):
    # Predict using the input model
    prediction_svc = svc_model.predict(create_features_from_text(text))[0]
    prediction_svc_proba = svc_model.predict_proba(create_features_from_text(text))[0]
    # Return result
    category_svc = get_category_name(prediction_svc)
    print("The predicted category using the SVM model is %s." % (category_svc))
    print("The conditional probability is: %a" % (prediction_svc_proba.max() * 100))
    print()


def predict_from_text_gbc(text):
    # Predict using the input model
    prediction_gbc = gbc_model.predict(create_features_from_text(text))[0]
    prediction_gbc_proba = gbc_model.predict_proba(create_features_from_text(text))[0]
    # Return result
    category_gbc = get_category_name(prediction_gbc)
    print("The predicted category using the GBC model is %s." % (category_gbc))
    print("The conditional probability is: %a" % (prediction_gbc_proba.max() * 100))
    print()


def predict_from_text_knnc(text):
    # Predict using the input model
    prediction_knnc = knnc_model.predict(create_features_from_text(text))[0]
    prediction_knnc_proba = knnc_model.predict_proba(create_features_from_text(text))[0]
    # Return result
    category_knnc = get_category_name(prediction_knnc)
    print("The predicted category using the KNNC model is %s." % (category_knnc))
    print("The conditional probability is: %a" % (prediction_knnc_proba.max() * 100))
    print()


def predict_from_text_lrc(text):
    # Predict using the input model
    prediction_lrc = lrc_model.predict(create_features_from_text(text))[0]
    prediction_lrc_proba = lrc_model.predict_proba(create_features_from_text(text))[0]
    # Return result
    category_lrc = get_category_name(prediction_lrc)
    print("The predicted category using the LRC model is %s." % (category_lrc))
    print("The conditional probability is: %a" % (prediction_lrc_proba.max() * 100))
    print()


def predict_from_text_mnbc(text):
    # Predict using the input model
    prediction_mnbc = mnbc_model.predict(create_features_from_text(text))[0]
    prediction_mnbc_proba = mnbc_model.predict_proba(create_features_from_text(text))[0]
    # Return result
    category_mnbc = get_category_name(prediction_mnbc)
    print("The predicted category using the MNBC model is %s." % (category_mnbc))
    print("The conditional probability is: %a" % (prediction_mnbc_proba.max() * 100))
    print()


def predict_from_text_rfc(text):
    # Predict using the input model
    prediction_rfc = rfc_model.predict(create_features_from_text(text))[0]
    prediction_rfc_proba = rfc_model.predict_proba(create_features_from_text(text))[0]
    # Return result
    category_rfc = get_category_name(prediction_rfc)
    print("The predicted category using the RFC model is %s." % (category_rfc))
    print("The conditional probability is: %a" % (prediction_rfc_proba.max() * 100))
    print()


with open('data/testArticle.txt', encoding="utf8") as f:
    lines = f.readlines()
text = ' '.join([str(elem) for elem in lines])

predict_from_text_svc(text)
predict_from_text_gbc(text)
predict_from_text_knnc(text)
predict_from_text_lrc(text)
predict_from_text_mnbc(text)
predict_from_text_rfc(text)

