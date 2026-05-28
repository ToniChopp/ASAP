import pandas as pd
import os

csv_path = "./RadGenome-ChestCT/dataset/radgenome_files/subset_2000/train_image_list.csv"

df = pd.read_csv(csv_path)
volume_list = df.iloc[:, 0]

volume_path = []

### TODO: Change volume_root to the path where you save the preprocessed CT volumes.
volume_root = "../../../Data/CT-Rate/train_preprocessed"
for i in range(len(volume_list)):
    volume_id = volume_list[i]
    paths = volume_id.split("_")
    path = paths[0] + "_" + paths[1] + "/"  + paths[0] + "_" + paths[1] + paths[2] + "/" + volume_id
    volume_path.append(os.path.join(volume_root, path))

df_out = pd.DataFrame({"Volumename": volume_list, "nii_path": volume_path})
df_out.to_csv("./RadGenome-ChestCT/dataset/radgenome_files/subset_2000/train_image_path.csv", index=False)