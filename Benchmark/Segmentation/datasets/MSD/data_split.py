import json
import numpy as np
import ipdb
from sklearn.model_selection import train_test_split

json_path = "./MSD_Task06_Lung_100.json"
with open(json_path, 'r') as f:
    json_data = json.load(f)

training_list = json_data['training']
# ipdb.set_trace()

train_list_10, _ = train_test_split(training_list, train_size=0.1, random_state=42)
json_data_10 = json_data.copy()
json_data_10['training'] = train_list_10
with open("./MSD_Task06_Lung_10.json", 'w') as f:
    json.dump(json_data_10, f, indent=4)