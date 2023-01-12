import glob
import os
import shutil
import random
from tqdm import tqdm
import nltk

# Vocabulary
ia_vocabulary = ['abduct', 'hijack', 'hostage', 'assault', 'sexually', 'torture', 'kill', 'bombing', 'assassinate']
if_vocabulary = ['blockade', 'restrict', 'occupy', 'fight', 'artillery', 'tanks', 'weapons', 'violate']
ip_vocabulary = ['engage', 'political', 'demonstrate', 'rally', 'leadership', 'rights', 'regime', 'strike', 'boycott',
                 'block', 'obstruct', 'demand', 'protest']
it_vocabulary = ['threaten', 'non-force', 'boycott', 'embargo', 'sanction', 'sanctions', 'restrictions', 'ban',
                 'impose',
                 'repression', 'blockade', 'occupation', 'ultimatum']
iu_vocabulary = ['engage', 'massive', 'expulsion', 'mass', 'killings', 'cleansing', 'nuclear', 'chemical', 'biological',
                 'radiological', 'detonate']

# Placeholders
ia_placeholder = []
if_placeholder = []
ip_placeholder = []
it_placeholder = []
iu_placeholder = []


def create_file_list(name, datapath, vocab, placeholder, final_spot):
    for filename in tqdm(glob.glob(os.path.join(datapath, '*.txt')), desc="Creating Folder for " + str(name)):
        with open(os.path.join(os.getcwd(), filename), 'r') as f:
            content = f.read().lower()
            words = nltk.word_tokenize(content)
            matching = [word for word in words if word in vocab]
            if len(matching) >= 1:
                placeholder.append(filename)
    i = 0
    if len(placeholder) < 400:
        numbers = random.sample(range(0, len(placeholder)), len(placeholder))
    else:
        numbers = random.sample(range(0, len(placeholder)), 400)
    for file in numbers:
        shutil.copy(placeholder[int(file)], final_spot)


create_file_list('Assault', 'data/gdelt/India_Assault/', ia_vocabulary, ia_placeholder, 'data/GoldenDataset2/Assault')
create_file_list('Fight', 'data/gdelt/India_Fight/', if_vocabulary, if_placeholder, 'data/GoldenDataset2/Fight')
create_file_list('Protest', 'data/gdelt/India_Protest/', ip_vocabulary, ip_placeholder, 'data/GoldenDataset2/Protest')
create_file_list('Threaten', 'data/gdelt/India_Threaten/', it_vocabulary, it_placeholder, 'data/GoldenDataset2/Threaten')
create_file_list('UMV', 'data/gdelt/India_UMV/', iu_vocabulary, iu_placeholder, 'data/GoldenDataset2/UMV')
print("Dataset Created!")

