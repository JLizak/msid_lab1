import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("80s/archive/UltimateClassicRock.csv")
df['Decade'] = (df['Year'] // 10) * 10
df['Decade'] = df['Decade'].astype(str) + 's'
order = ['1960s', '1970s', '1980s', '1990s', '2000s', '2010s', '2020s']
df['Decade'] = pd.Categorical(df['Decade'], categories=order, ordered=True)


def convert_duration(duration):
    minutes, seconds = map(int, duration.split(':'))
    return minutes * 60 + seconds

df['Duration'] = df['Duration'].apply(convert_duration)



data = df.describe(percentiles=[0.05, 0.95])

missing_values = df.isna().sum()
data = data.drop("count", axis=0)
data.loc["missing_values"] = missing_values

data.to_csv("numericalStatistics.csv", index=False)


categorical_cols = df.select_dtypes(include=['object', 'category']).columns

desc = df[categorical_cols].describe()

desc.loc['missing'] = df[categorical_cols].isnull().sum()

desc.loc['class_proportions'] = [
    df[col].value_counts(normalize=True).round(2).to_dict()
    for col in categorical_cols
]

desc.drop(index=["top", "freq", "count"])
data.to_csv("categoricalStatistics.csv", index=False)



plt.figure(figsize=(12, 6))
sns.boxplot(x=df["Year"], y=df["Danceability"])
plt.xticks(rotation=45)
plt.savefig("plt1.png", dpi=300, bbox_inches='tight')


plt.figure(figsize=(12, 6))
sns.boxplot(x=df["Year"], y=df["Tempo"])
plt.xticks(rotation=45)
plt.savefig("plt2.png", dpi=300, bbox_inches='tight')


plt.figure(figsize=(12, 6))
sns.boxplot(x=df["Year"], y=df["Popularity"])
plt.xticks(rotation=45)
plt.savefig("plt3.png", dpi=300, bbox_inches='tight')


sns.violinplot(x='Decade', y='Popularity', data=df)
plt.xlabel("Dekada")
plt.ylabel("Popularity")
plt.savefig("plt4.png", dpi=300, bbox_inches='tight')


sns.violinplot(x='Decade', y='Energy', data=df)
plt.xlabel("Decade")
plt.ylabel("Energy")
plt.savefig("plt5.png", dpi=300, bbox_inches='tight')


plt.figure(figsize=(12, 6))
sns.pointplot(x='Year', y='Loudness', data=df, errorbar='sd')
plt.xlabel("Year")
plt.ylabel("Loudness")

years = sorted(df['Year'].unique())
plt.xticks(ticks=range(0, len(years), 2), rotation=45, labels=years[::2])
plt.savefig("plt6.png", dpi=300, bbox_inches='tight')


plt.figure(figsize=(12, 6))
sns.pointplot(x='Year', y='Energy', data=df, errorbar='sd')
plt.xlabel("Year")
plt.ylabel("Energy")

years = sorted(df['Year'].unique())
plt.xticks(ticks=range(0, len(years), 2), rotation=45, labels=years[::2])
plt.savefig("plt7.png", dpi=300, bbox_inches='tight')



hists = plt.figure(figsize=(14, 10))

ax1 = hists.add_subplot(2, 2, 1)
ax2 = hists.add_subplot(2, 2, 2)
ax3 = hists.add_subplot(2, 2, 3)
ax4 = hists.add_subplot(2, 2, 4)

sns.histplot(df, x="Year", ax=ax1)
sns.histplot(df, x="Popularity", bins=30, ax=ax2)
sns.histplot(df, x="Danceability", ax=ax3)
sns.histplot(df, x="Tempo", bins=50, ax=ax4)
plt.savefig("plt8.png", dpi=300, bbox_inches='tight')


hists = plt.figure(figsize=(14, 10))

ax1 = hists.add_subplot(2, 2, 1)
ax2 = hists.add_subplot(2, 2, 2)

sns.histplot(df, x="Danceability", hue="Time_Signature", palette="Set2", ax=ax1)
sns.histplot(df, x="Popularity", bins=50, hue="Decade", palette="Set2", ax=ax2)
plt.savefig("plt9.png", dpi=300, bbox_inches='tight')


artist_track_count = df['Artist'].value_counts()
top_20_artists = artist_track_count.nlargest(20)

plt.figure(figsize=(12, 5))
plt.xticks(rotation=45)

top_20_df = top_20_artists.reset_index()
top_20_df.columns = ['Artist', 'Track Count']

sns.barplot(top_20_df, x="Artist", y="Track Count")
plt.savefig("plt9.png", dpi=300, bbox_inches='tight')


plt.figure(figsize=(8, 8))
features = ["Year", "Loudness", "Danceability", "Liveness", "Valence", "Acousticness", "Tempo", "Energy", "Popularity"]
sns.heatmap(df[features].corr(), annot=True, fmt=".2f")
plt.savefig("plt10.png", dpi=300, bbox_inches='tight')


df_80s = df[df['Decade'] == "1980s"]

sns.regplot(x="Energy", y="Loudness", data=df_80s)
plt.savefig("plt11.png", dpi=300, bbox_inches='tight')


sns.regplot(x="Danceability", y="Valence", data=df_80s)
plt.savefig("plt12.png", dpi=300, bbox_inches='tight')


sns.regplot(x="Acousticness", y="Energy", data=df_80s)
plt.savefig("plt13.png", dpi=300, bbox_inches='tight')






