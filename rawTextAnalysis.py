# TODO: Any graph you do not want to visualize you can comment out that set of code

# TODO: Find a way to save the figures to a folder
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

sns.set_style("whitegrid")
import altair as alt

# alt.renderers.enable("notebook")

# Code for hiding seaborn warnings
import warnings

warnings.filterwarnings("ignore")

# Load Dataset
# TODO: PLACE DATASET LOCATION/DATASET FILE HERE **Must be a CSV FILE**
df_path = "D:/Graduate School/text-classifier/data/"
df_path2 = df_path + 'News_India_dataset.csv'
df = pd.read_csv(df_path2, sep=';')

# Create a graph to see how many articles exist per category
bars = alt.Chart(df).mark_bar(size=50).encode(
    x=alt.X("Category"),
    y=alt.Y("count():Q", axis=alt.Axis(title='Number of articles')),
    tooltip=[alt.Tooltip('count()', title='Number of articles'), 'Category'],
    color='Category'

)

text = bars.mark_text(
    align='center',
    baseline='bottom',
).encode(
    text='count()'
)

(bars + text).interactive().properties(
    height=300,
    width=700,
    title="Number of articles in each category",
)

# Create the same graph but display the percentage of categories in relation to the rest of the dataset
df['id'] = 1
df2 = pd.DataFrame(df.groupby('Category').count()['id']).reset_index()

bars = alt.Chart(df2).mark_bar(size=50).encode(
    x=alt.X('Category'),
    y=alt.Y('PercentOfTotal:Q', axis=alt.Axis(format='.0%', title='% of Articles')),
    color='Category'
).transform_window(
    TotalArticles='sum(id)',
    frame=[None, None]
).transform_calculate(
    PercentOfTotal="datum.id / datum.TotalArticles"
)

text = bars.mark_text(
    align='center',
    baseline='bottom',
    # dx=5  # Nudges text to right so it doesn't appear on top of the bar
).encode(
    text=alt.Text('PercentOfTotal:Q', format='.1%')
)

(bars + text).interactive().properties(
    height=300,
    width=700,
    title="% of articles in each category",
)

# Graph news length distribution
df['News_length'] = df['Content'].str.len()
plt.figure(figsize=(12.8, 6))

sns.distplot(df['News_length']).set_title('News length distribution')
plt.savefig("images/india_dataset/dataset_bar.png")
# Some analysis on the news data [removes outliers] **MAY NEED TO CHANGE THE QUARTILE**
df['News_length'].describe()
quantile_95 = df['News_length'].quantile(0.95)
df_95 = df[df['News_length'] < quantile_95]

# Graph the new data set based off the removed outliers
plt.figure(figsize=(12.8, 6))
sns.distplot(df_95['News_length']).set_title('News length distribution')
plt.savefig("images/india_dataset/dataset_bar_updated.png")
# See how many articles contain over 10000 characters
df_more10k = df[df['News_length'] > 10000]
print(len(df_more10k))

# TODO: uncomment if you want to see some of the content that is over 10000
# df_more10k['Content'].iloc[0]

# Graph full set based off of article length
plt.figure(figsize=(12.8, 6))
sns.boxplot(data=df, x='Category', y='News_length', width=.5)
plt.savefig("images/india_dataset/full_set.png")
# Graph set off reduced article length [REMEMBER ALL OF THIS CAN BE CHANGED]
plt.figure(figsize=(12.8, 6))
sns.boxplot(data=df_95, x='Category', y='News_length')
plt.savefig("images/india_dataset/95_set.png")
# Save the dataset as a pickle to begin the raw text analysis to prepare for training
# TODO: REPLACE FILE NAME AND LOCATION
with open('data/News_india_dataset.pickle', 'wb') as output:
    pickle.dump(df, output)
