import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

df_tracks = pd.read_csv(
    "/run/media/high/Edu/Music Genre Classification Project/fma_metadata/tracks.csv",
    low_memory=False,
)

df = df_tracks[["track_id", "subset", "bit_rate", "duration", "genre_top", "title"]]

df_small = df[df["subset"] == "small"].copy()

df_small["track_id"] = df_small["track_id"].apply(str)
df_small["track_id"] = list(map(lambda x: x.zfill(6), df_small["track_id"]))
df_small["file_name"] = df_small["track_id"] + ".mp3"

genre_list = df_small["genre_top"].unique().tolist()

le = preprocessing.LabelEncoder()
le.fit(genre_list)
df_small["genre_class"] = df_small["genre_top"].apply(lambda x: le.transform([x])[0])

df_train, df_test = train_test_split(df_small, test_size=0.2)
df_train["train-test"] = "Train"
df_test["train-test"] = "Test"
result = pd.concat([df_train, df_test])
result.sort_index()


result = result[
    [
        "file_name",
        "track_id",
        "subset",
        "bit_rate",
        "duration",
        "genre_top",
        "genre_class",
        "train-test",
        "title",
    ]
]
result.sort_values(by="track_id", ignore_index=True, inplace=True)

result.to_csv(
    "/run/media/high/Edu/Music Genre Classification Project/Music-Genre-Recognition/track-genre.csv",
    index=False,
)

print(result.info())
print(result["train-test"].value_counts())
print(sorted(genre_list))
print(le.transform(sorted(genre_list)))
