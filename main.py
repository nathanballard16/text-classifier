import pandas

from RuleGeneration import run_rule_generation, create_applier, save_training_data
from svm import train_svm
from knn import train_knn

labels = ['protest_path', 'threaten_path', 'umv_path', 'assault_path']
labeled_file = ['protest_file', 'threaten_file', 'umv_file', 'assault_file']
golden_dataset_labels = ['orig_protest', 'orig_threaten', 'orig_umv', 'orig_assault']
category_codes = {
    'Assault': 0,
    'Protest': 1,
    'Threaten': 2,
    'Umv': 3
}
# category_codes = {
#     'Assault': 0,
#     'Fight': 1,
#     'Protest': 2,
#     'Threaten': 3,
#     'Umv': 4
# }
train_path = './data/GDELT_Labeled/train_test/'
model_out = './data/GDELT_labeled/model_out/'


def train_golden_set():
    pass


def generate_applier():
    return create_applier()


def generate_snorkel_labeled_datasets(lfs, applier):
    df = pandas.DataFrame()
    for label in range(len(labels)):
        df2 = run_rule_generation(labels[label], labeled_file[label], golden_dataset_labels[label], lfs, applier)
        # print(df2)
        df = df.append(df2, ignore_index=True)
    # print(df)
    return df


def train_linear_models():
    print()
    print('*************TRAINING MODELS*************')
    print()
    # Train SVM
    # train_svm(train_path, model_out)
    train_knn(train_path, model_out)


def main():
#     lfs, applier = generate_applier()
#     df = generate_snorkel_labeled_datasets(lfs, applier)
#     save_training_data(df, category_codes)
    train_linear_models()


if __name__ == '__main__':
    main()
