import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import ipdb

# csv_file = "./CT-Rate train.csv"
# df = pd.read_csv(csv_file)

# kf = KFold(n_splits=5, shuffle=True, random_state=42)

# for fold, (train_index, test_index) in enumerate(kf.split(df)):
#     df_train = df.iloc[train_index]
#     df_test = df.iloc[test_index]

#     kf_val_test = KFold(n_splits=2, shuffle=True, random_state=42)
#     val_index, test_index = kf_val_test.split(df_test)
#     # ipdb.set_trace()
#     df_val = df_test.iloc[val_index[0]]
#     df_test = df_test.iloc[test_index[0]]

#     df_train.to_csv(f"./train_list_{fold+1}.csv", index=False)
#     df_val.to_csv(f"./val_list_{fold+1}.csv", index=False)
#     df_test.to_csv(f"./test_list_{fold+1}.csv", index=False)

# df_train, df_val = train_test_split(df, test_size=0.1, random_state=42)
# df_train.to_csv(f"./train_list.csv", index=False)
# df_val.to_csv(f"./val_list.csv", index=False)



# # random split for 1% and 10% of training data
# csv_file = "LUNA16.csv"
# df = pd.read_csv(csv_file)

# df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)
# df_train.to_csv("./train_list.csv", index=False)
# df_test.to_csv("./test_list.csv", index=False)
# df_test.to_csv("./val_list.csv", index=False)

# df_train_1 = df_train.sample(frac=0.01, random_state=42)
# df_train_1.to_csv("./train_list_1.csv", index=False)
# df_train_10 = df_train.sample(frac=0.1, random_state=42)
# df_train_10.to_csv("./train_list_10.csv", index=False)

df = pd.read_csv("train_list.csv")

df_train_1 = df.sample(frac=0.01, random_state=42)
df_train_1.to_csv("train_list_1.csv", index=False)


