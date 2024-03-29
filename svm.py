import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def train_svm(input_path, output_path):
    # Dataframe
    path_df = input_path + "df.pickle"
    with open(path_df, 'rb') as data:
        df = pickle.load(data)

    # features_train
    path_features_train = input_path + "features_train.pickle"
    with open(path_features_train, 'rb') as data:
        features_train = pickle.load(data)

    # labels_train
    path_labels_train = input_path + "labels_train.pickle"
    with open(path_labels_train, 'rb') as data:
        labels_train = pickle.load(data)

    # features_test
    path_features_test = input_path + "features_test.pickle"
    with open(path_features_test, 'rb') as data:
        features_test = pickle.load(data)

    # labels_test
    path_labels_test = input_path + "labels_test.pickle"
    with open(path_labels_test, 'rb') as data:
        labels_test = pickle.load(data)

    # print(features_train.shape)
    # print(features_test.shape)
    # print(np.unique(features_test))
    svc_0 = svm.SVC(random_state=8)

    # svc_0.fit(features_train, labels_train)
    #
    # # print prediction results
    # predictions = svc_0.predict(features_test)
    # print(classification_report(labels_test, predictions))

    print('Parameters currently in use:\n')
    pprint(svc_0.get_params())

    # # defining parameter range
    # param_grid = {'C': [0.1, 1, 10, 100, 1000],
    #               'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    #               'kernel': ['rbf']}
    #
    # grid = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=3)
    #
    # # fitting the model for grid search
    # grid.fit(features_train, labels_train)
    #
    # print()
    #
    # # print best parameter after tuning
    # print(grid.best_params_)
    #
    # # print how our model looks after hyper-parameter tuning
    # print(grid.best_estimator_)
    #
    # print()
    # grid_predictions = grid.predict(features_test)
    #
    # # print classification report
    # print(classification_report(labels_test, grid_predictions))
    # exit()

    # C
    # C = [.0001, .001, .01]
    C = [0.1, 1, 10, 100, 1000]

    # gamma
    # gamma = [.0001, .001, .01, .1, 1, 10, 100]
    gamma = [.0001, .001, .01, .1, 1]

    # degree
    degree = [1, 2, 3, 4, 5]

    # kernel
    kernel = ['linear', 'rbf', 'poly']

    # probability
    probability = [True]

    # Create the random grid
    random_grid = {'C': C,
                   'kernel': kernel,
                   'gamma': gamma,
                   'degree': degree,
                   'probability': probability
                   }

    pprint(random_grid)

    # First create the base model to tune
    svc = svm.SVC(random_state=8)

    # Definition of the random search
    random_search = RandomizedSearchCV(estimator=svc,
                                       param_distributions=random_grid,
                                       n_iter=50,
                                       scoring='accuracy',
                                       cv=3,
                                       verbose=1,
                                       random_state=8)

    # Fit the random search model
    random_search.fit(features_train, labels_train)

    print("The best hyperparameters from Random Search are:")
    print(random_search.best_params_)
    print("")
    print("The mean accuracy of a model with these hyperparameters is:")
    print(random_search.best_score_)

    # Create the parameter grid based on the results of random search
    C = [100, 1000, 10000, 10000]
    degree = [4, 5, 6]
    gamma = [.1, 1, 10]
    probability = [True]

    param_grid = [
        {'C': C, 'kernel': ['linear'], 'probability': probability},
        {'C': C, 'kernel': ['poly'], 'degree': degree, 'probability': probability},
        {'C': C, 'kernel': ['rbf'], 'gamma': gamma, 'probability': probability}
    ]

    # Create a base model
    svc = svm.SVC(random_state=8)

    # Manually create the splits in CV in order to be able to fix a random_state (GridSearchCV doesn't have that
    # argument)
    cv_sets = ShuffleSplit(n_splits=3, test_size=.33, random_state=8)

    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=svc,
                               param_grid=param_grid,
                               scoring='accuracy',
                               cv=cv_sets,
                               verbose=1)

    # Fit the grid search to the data
    grid_search.fit(features_train, labels_train)

    print("The best hyperparameters from Grid Search are:")
    print(grid_search.best_params_)
    print("")
    print("The mean accuracy of a model with these hyperparameters is:")
    print(grid_search.best_score_)

    best_svc = grid_search.best_estimator_
    print(best_svc)

    best_svc.fit(features_train, labels_train)
    svc_pred = best_svc.predict(features_test)
    # Training accuracy
    print("The training accuracy is: ")
    print(accuracy_score(labels_train, best_svc.predict(features_train)))
    # Test accuracy
    print("The test accuracy is: ")
    print(accuracy_score(labels_test, svc_pred))
    # Classification report
    print("Classification report")
    print(classification_report(labels_test, svc_pred))

    # Confusion Matrix
    aux_df = df[['Category', 'Category_Code']].drop_duplicates().sort_values('Category_Code')
    conf_matrix = confusion_matrix(labels_test, svc_pred)
    plt.figure(figsize=(12.8, 6))
    sns.heatmap(conf_matrix,
                annot=True,
                xticklabels=aux_df['Category'].values,
                yticklabels=aux_df['Category'].values,
                cmap="Blues")
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    plt.title('Confusion matrix')
    plt.show()

    base_model = svm.SVC(random_state=8)
    base_model.fit(features_train, labels_train)
    accuracy_score(labels_test, base_model.predict(features_test))

    best_svc.fit(features_train, labels_train)
    accuracy_score(labels_test, best_svc.predict(features_test))
    d = {
        'Model': 'SVM',
        'Training Set Accuracy': accuracy_score(labels_train, best_svc.predict(features_train)),
        'Test Set Accuracy': accuracy_score(labels_test, svc_pred)
    }

    df_models_svc = pd.DataFrame(d, index=[0])
    print(df_models_svc)

    with open(output_path + 'best_svc.pickle', 'wb') as output:
        pickle.dump(best_svc, output)

    with open(output_path + 'df_models_svc.pickle', 'wb') as output:
        pickle.dump(df_models_svc, output)
