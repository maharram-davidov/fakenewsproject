import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
from wordcloud import WordCloud

# Load dataset
fake = pd.read_csv("../data/Fake.csv")
true = pd.read_csv("../data/True.csv")

fake["label"] = 1
true["label"] = 0

df = pd.concat([fake, true])

print("Dataset shape:", df.shape)
print(df.head())

# Label distribution
print("\nLabel distribution:")
print(df["label"].value_counts())

# Plot Fake vs Real distribution
plt.figure()
df["label"].value_counts().plot(kind="bar")
plt.xticks([0,1],["Real","Fake"])
plt.title("Fake vs Real News Distribution")
plt.show()

# Text length analysis
df["text_length"] = df["text"].apply(len)

print("\nText length statistics:")
print(df["text_length"].describe())

# Histogram of text length
plt.figure()
sns.histplot(df[df["label"]==1]["text_length"], label="Fake", kde=True)
sns.histplot(df[df["label"]==0]["text_length"], label="Real", kde=True)

plt.legend()
plt.title("Text Length Distribution")
plt.show()

# Most common words
all_text = " ".join(df["text"]).lower()

words = re.findall(r'\b[a-z]+\b', all_text)

word_counts = Counter(words)

print("\nMost common words:")
print(word_counts.most_common(20))

# Wordcloud
wordcloud = WordCloud(width=800, height=400).generate(all_text)

plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Word Cloud of News Articles")
plt.show()

# Fake news word analysis
fake_text = " ".join(df[df["label"]==1]["text"]).lower()

fake_words = re.findall(r'\b[a-z]+\b', fake_text)

fake_counts = Counter(fake_words)

print("\nMost common words in Fake News:")
print(fake_counts.most_common(20))