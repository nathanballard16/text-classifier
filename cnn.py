from __future__ import print_function
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from keras.models import load_model
import warnings
import pickle

warnings.filterwarnings('ignore')
pd.set_option("display.max_rows", None, "display.max_columns", None)

path_df = "data/GDELT_Labeled/train_test_100/df.pickle"
with open(path_df, 'rb') as data:
    df = pickle.load(data)

# print(df)
# Change when new labels are applied
# category_id = []
# for value in df["Category"]:
#     if value == 'Protest':
#         category_id.append('1')
#     elif value == 'UMV':
#         category_id.append('2')
#     elif value == 'Assault':
#         category_id.append('0')
#
# df["Category_Code"] = category_id
# df['category_id'] = df.Category.factorize()[0]


def preprocess_text(sen):
    # Lowercase
    sentence = sen.lower()

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Zäöüß]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence


df.Content = df.Content.apply(preprocess_text)
print(df[::100])

# set parameters:
max_features = 20000
maxlen = 400
batch_size = 25
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 5


X = df['Content'].values
y = df['Category_Code']
y = to_categorical(y, num_classes=3)
print(y.shape)

print('Loading data...')
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(x_train)

x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)

x_train = pad_sequences(x_train, padding='post', maxlen=maxlen)
x_test = pad_sequences(x_test, padding='post', maxlen=maxlen)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print('Pad sequences (samples x time)')

x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)

import json

tokenizer_json = tokenizer.to_json()
with open('models/cnn_model/tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=True))

print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))
model.add(Dropout(0.2))

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
# we use max pooling:
model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(3))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

plot_model(model, to_file='images/cnn_model/CNN-model_plot.png', show_shapes=True, show_layer_names=True, dpi=100)

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    callbacks=[EarlyStopping(patience=3, monitor='val_loss', min_delta=0.0001)])

loss, Accuracy = model.evaluate(x_test, y_test)

print("Test Loss:", loss)
print("Test Accuracy:", Accuracy)

import matplotlib.pyplot as plt


# plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    plt.savefig('images/cnn_model/training_and_val_acc.png')


plot_history(history)

model.save('data/GDELT_Labeled/model_out_100/CNN.h5')
model.save_weights('data/GDELT_Labeled/model_out_100/CNN_weights.h5')
