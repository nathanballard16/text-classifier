import yaml
from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model.label_model import LabelModel



def read_config():
    # Reading in the YAML config file
    print("Reading Config File ...")
    with open("Snorkel_config/Snorkel_Config.yaml", "r") as yamlfile:
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)
        print("Read Config File Successful")
    return data


def label_data():
    ABSTAIN = -1
    LABELED = 1
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

def main():
    data = read_config()
    print(data)


if __name__ == "__main__":
    main()
