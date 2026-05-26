import json
import numpy as np
import ipdb
from sklearn.model_selection import train_test_split

json_path = "./dataset.json"
with open(json_path, 'r') as f:
    json_data = json.load(f)

training_list = json_data['training']
val_list = json_data['validation']
test_list = json_data['testing']
# ipdb.set_trace()

training_list = sorted(training_list, key=lambda x: x["image"])
val_list = sorted(val_list, key=lambda x: x["image"])
test_list = sorted(test_list, key=lambda x: x["image"])

json_data['training'] = training_list
json_data['validation'] = val_list
json_data['testing'] = test_list

with open("./dataset.json", 'w') as f:
    json.dump(json_data, f, indent=4)

# train_list_10, _ = train_test_split(training_list, train_size=0.1, random_state=42)
# json_data_10 = json_data.copy()
# json_data_10['training'] = train_list_10
# with open("./dataset_10.json", 'w') as f:
#     json.dump(json_data_10, f, indent=4)