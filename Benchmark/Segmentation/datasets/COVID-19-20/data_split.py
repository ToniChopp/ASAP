import json
import numpy as np
import ipdb
from sklearn.model_selection import train_test_split

json_path = "./dataset_100.json"
with open(json_path, 'r') as f:
    json_data = json.load(f)

training_list = json_data['training']
# ipdb.set_trace()

train_list_10, _ = train_test_split(training_list, train_size=0.2, random_state=42)
json_data_10 = json_data.copy()
json_data_10['training'] = train_list_10
with open("./dataset_20.json", 'w') as f:
    json.dump(json_data_10, f, indent=4)

# with open(json_file, "r") as f:
#     data = json.load(f)

# train_list = data["training"]
# test_list = data["testing"]
# val_list = data["validation"]

# for i in range(len(train_list)):
#     train_list[i]["label"] = train_list[i]["image"].replace("_ct", "_seg").replace("imagesTr", "labelsTr")

# for i in range(len(test_list)):
#     test_list[i]["label"] = test_list[i]["image"].replace("_ct", "_seg").replace("imagesTr", "labelsTr")
# for i in range(len(val_list)):
#     val_list[i]["label"] = val_list[i]["image"].replace("_ct", "_seg").replace("imagesTr", "labelsTr")

# data["training"] = train_list
# data["testing"] = test_list
# data["validation"] = val_list

# with open(json_file, "w") as f:
#     json.dump(data, f, indent=4)