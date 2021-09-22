import pickle
import pandas as pd
import numpy as np
import random

# Dataframe
path_df = "train_test/df.pickle"
with open(path_df, 'rb') as data:
    df = pickle.load(data)

# X_train
path_X_train = "train_test/X_train.pickle"
with open(path_X_train, 'rb') as data:
    X_train = pickle.load(data)

# X_test
path_X_test = "train_test/X_test.pickle"
with open(path_X_test, 'rb') as data:
    X_test = pickle.load(data)

# y_train
path_y_train = "train_test/y_train.pickle"
with open(path_y_train, 'rb') as data:
    y_train = pickle.load(data)

# y_test
path_y_test = "train_test/y_test.pickle"
with open(path_y_test, 'rb') as data:
    y_test = pickle.load(data)

# features_train
path_features_train = "train_test/features_train.pickle"
with open(path_features_train, 'rb') as data:
    features_train = pickle.load(data)

# labels_train
path_labels_train = "train_test/labels_train.pickle"
with open(path_labels_train, 'rb') as data:
    labels_train = pickle.load(data)

# features_test
path_features_test = "train_test/features_test.pickle"
with open(path_features_test, 'rb') as data:
    features_test = pickle.load(data)

# labels_test
path_labels_test = "train_test/labels_test.pickle"
with open(path_labels_test, 'rb') as data:
    labels_test = pickle.load(data)

# SVM Model
path_model = "models/best_svc.pickle"
with open(path_model, 'rb') as data:
    svc_model = pickle.load(data)

# Category mapping dictionary
category_codes = {
    'business': 0,
    'entertainment': 1,
    'politics': 2,
    'sport': 3,
    'tech': 4
}

category_names = {
    0: 'business',
    1: 'entertainment',
    2: 'politics',
    3: 'sport',
    4: 'tech'
}

predictions = svc_model.predict(features_test)
# Indexes of the test set
index_X_test = X_test.index

# We get them from the original df
df_test = df.loc[index_X_test]

# Add the predictions
df_test['Prediction'] = predictions

# Clean columns
df_test = df_test[['Content', 'Category', 'Category_Code', 'Prediction']]

# Decode
df_test['Category_Predicted'] = df_test['Prediction']
df_test = df_test.replace({'Category_Predicted':category_names})

# Clean columns again
df_test = df_test[['Content', 'Category', 'Category_Predicted']]
print(df_test.head())

condition = (df_test['Category'] != df_test['Category_Predicted'])

df_misclassified = df_test[condition]

print(df_misclassified.head(3))

def output_article(row_article):
    print('Actual Category: %s' %(row_article['Category']))
    print('Predicted Category: %s' %(row_article['Category_Predicted']))
    print('-------------------------------------------')
    print('Text: ')
    print('%s' %(row_article['Content']))

random.seed(8)
list_samples = random.sample(list(df_misclassified.index), 3)
print(list_samples)
print("First Case:")
print(output_article(df_misclassified.loc[list_samples[0]]))
print("Second Case:")
print(output_article(df_misclassified.loc[list_samples[1]]))
print("Third Case:")
print(output_article(df_misclassified.loc[list_samples[2]]))



