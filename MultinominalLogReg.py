import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def train_mlr(input_path, output_path):
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

    lr_0 = LogisticRegression(random_state = 8)

    print('Parameters currently in use:\n')
    pprint(lr_0.get_params())

    # C
    C = [float(x) for x in np.linspace(start = 0.1, stop = 1, num = 10)]

    # multi_class
    multi_class = ['multinomial']

    # solver
    solver = ['newton-cg', 'sag', 'saga', 'lbfgs']

    # class_weight
    class_weight = ['balanced', None]

    # penalty
    penalty = ['l2']

    # Create the random grid
    random_grid = {'C': C,
                   'multi_class': multi_class,
                   'solver': solver,
                   'class_weight': class_weight,
                   'penalty': penalty}

    pprint(random_grid)

    # First create the base model to tune
    lrc = LogisticRegression(random_state=8)

    # Definition of the random search
    random_search = RandomizedSearchCV(estimator=lrc,
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
    C = [float(x) for x in np.linspace(start = 0.6, stop = 1, num = 10)]
    multi_class = ['multinomial']
    solver = ['sag']
    class_weight = ['balanced']
    penalty = ['l2']

    param_grid = {'C': C,
                  'multi_class': multi_class,
                  'solver': solver,
                  'class_weight': class_weight,
                  'penalty': penalty}

    # Create a base model
    lrc = LogisticRegression(random_state=8)

    # Manually create the splits in CV in order to be able to fix a random_state (GridSearchCV doesn't have that argument)
    cv_sets = ShuffleSplit(n_splits = 3, test_size = .33, random_state = 8)

    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=lrc,
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

    best_lrc = grid_search.best_estimator_
    print(best_lrc)

    best_lrc.fit(features_train, labels_train)
    lrc_pred = best_lrc.predict(features_test)

    # Training accuracy
    print("The training accuracy is: ")
    print(accuracy_score(labels_train, best_lrc.predict(features_train)))
    # Test accuracy
    print("The test accuracy is: ")
    print(accuracy_score(labels_test, lrc_pred))

    # Classification report
    print("Classification report")
    print(classification_report(labels_test,lrc_pred))

    # Confusion Matrix
    aux_df = df[['Category', 'Category_Code']].drop_duplicates().sort_values('Category_Code')
    conf_matrix = confusion_matrix(labels_test, lrc_pred)
    plt.figure(figsize=(12.8,6))
    sns.heatmap(conf_matrix,
                annot=True,
                xticklabels=aux_df['Category'].values,
                yticklabels=aux_df['Category'].values,
                cmap="Blues")
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    plt.title('Confusion matrix')
    plt.show()

    base_model = LogisticRegression(random_state = 8)
    base_model.fit(features_train, labels_train)
    accuracy_score(labels_test, base_model.predict(features_test))

    best_lrc.fit(features_train, labels_train)
    accuracy_score(labels_test, best_lrc.predict(features_test))
    d = {
        'Model': 'Logistic Regression',
        'Training Set Accuracy': accuracy_score(labels_train, best_lrc.predict(features_train)),
        'Test Set Accuracy': accuracy_score(labels_test, lrc_pred)
    }

    df_models_lrc = pd.DataFrame(d, index=[0])

    print(df_models_lrc)

    with open(output_path + 'best_lrc.pickle', 'wb') as output:
        pickle.dump(best_lrc, output)

    with open(output_path + 'df_models_lrc.pickle', 'wb') as output:
        pickle.dump(df_models_lrc, output)