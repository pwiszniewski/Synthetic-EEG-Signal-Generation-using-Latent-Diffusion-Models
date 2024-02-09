import pandas as pd
from sklearn.model_selection import train_test_split

file_path = "/home/bru/PycharmProjects/DDPM-EEG/data/ids_shhs/ids_shhs.csv"
file_train = file_path.replace(".csv", "_train.csv")
file_valid = file_path.replace(".csv", "_valid.csv")
file_test = file_path.replace(".csv", "_test.csv")

df = pd.read_csv(file_path)

train, test = train_test_split(df['subject'].unique(), test_size=0.2, random_state=42)
train, valid = train_test_split(train, test_size=0.25, random_state=42)

train_ids = df[df['subject'].isin(train)].reset_index(drop=True)
valid_ids = df[df['subject'].isin(valid)].reset_index(drop=True)
test_ids = df[df['subject'].isin(test)].reset_index(drop=True)

train_ids.to_csv(file_train, index=False)
valid_ids.to_csv(file_valid, index=False)
test_ids.to_csv(file_test, index=False)
print("Done")