import pandas as pd
import json

train_df = pd.read_csv("train_list.csv")
train_img_list = train_df["image_id"]
test_df = pd.read_csv("test_list.csv")
test_img_list = test_df["image_id"]

out_dict = {}
out_dict["description"] = "LUNA16"
labels = {}
labels["0"] = "background"
labels["1"] = "left lung"
labels["2"] = "right lung"
labels["3"] = "trachea"
out_dict["labels"] = labels
out_dict["licence"] = "yt"
out_dict["modality"] = {"0": "CT"}
out_dict["name"] = "LUNA16"
out_dict["numTest"] = 267
out_dict["numTraining"] = 621
out_dict["tensorImageSize"] = "3D"
training_dict = []
for i in range(len(train_img_list)):
    pair = {}
    pair["image"] = "images/" + train_img_list[i]
    pair["label"] = "masks_processed/" + train_img_list[i]
    training_dict.append(pair)
out_dict["training"] = training_dict
testing_dict = []
for i in range(len(test_img_list)):
    pair = {}
    pair["image"] = "images/" + test_img_list[i]
    pair["label"] = "masks_processed/" + test_img_list[i]
    testing_dict.append(pair)
out_dict["testing"] = testing_dict
val_dict = []
for i in range(int(len(test_img_list))):
    pair = {}
    pair["image"] = "images/" + test_img_list[i]
    pair["label"] = "masks_processed/" + test_img_list[i]
    val_dict.append(pair)
out_dict["validation"] = val_dict

with open("dataset_100.json", "w") as f:
    json.dump(out_dict, f, indent=4)