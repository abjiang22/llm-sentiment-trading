import pandas as pd

# Load Excel (adjust path as needed)
df = pd.read_csv("loughran-mcdonald.csv")

# Filter only rows marked as Positive or Negative
positive_words = set(df[df['Positive'] > 0]['Word'].str.lower())
negative_words = set(df[df['Negative'] > 0]['Word'].str.lower())

# Save to text file (optional)
with open("lm_positive.txt", "w") as f:
    f.write("\n".join(sorted(positive_words)))
with open("lm_negative.txt", "w") as f:
    f.write("\n".join(sorted(negative_words)))