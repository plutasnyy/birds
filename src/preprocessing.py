import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("birdclef-2022/train_metadata.csv")

encoder = LabelEncoder()
df["primary_label_encoded"] = encoder.fit_transform(df["primary_label"])

skf = StratifiedKFold(n_splits=5)
for k, (_, val_ind) in enumerate(skf.split(X=df, y=df["primary_label_encoded"])):
    df.loc[val_ind, "fold"] = k

print(df.head())
df.to_csv("birdclef-2022/train_metadata_new.csv", index=False)
