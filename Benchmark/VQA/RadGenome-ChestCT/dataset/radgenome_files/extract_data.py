import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# csv_path = "./subset_200/train_case_disorders.csv"
# df = pd.read_csv(csv_path)

# volume_list = df['Volumename'].tolist()
# nii_path_list = []
# for i in tqdm(range(len(volume_list))):
#     volume_name = str(volume_list[i])
#     paths = volume_name.split("_")
#     volume_path = paths[0] + "_" + paths[1] + "/"  + paths[0] + "_" + paths[1] + paths[2] + "/" + volume_name
#     volume_path = os.path.join("../../../../../Data/CT-Rate" + "/train_preprocessed", volume_path)
#     nii_path_list.append(volume_path)

# df_out = pd.DataFrame({"Volumename": volume_list, "nii_path": nii_path_list})
# df_out.to_csv("./subset_200/train_image_path.csv", index=False)



csv_path = "./subset_200/train_image_path.csv"
df = pd.read_csv(csv_path)
volume_list = df['Volumename'].tolist()


abnormality_path = "./subset_2000/train_vqa_abnormality.csv"
abnormality_df = pd.read_csv(abnormality_path)
abnormality_df_out = abnormality_df[abnormality_df['Volumename'].isin(volume_list)]
abnormality_df_out.to_csv("./subset_200/train_vqa_abnormality.csv", index=False)


location_path = "./subset_2000/train_vqa_location.csv"
location_df = pd.read_csv(location_path)
location_df_out = location_df[location_df['Volumename'].isin(volume_list)]
location_df_out.to_csv("./subset_200/train_vqa_location.csv", index=False)


presence_path = "./subset_2000/train_vqa_presence.csv"
presence_df = pd.read_csv(presence_path)
presence_df_out = presence_df[presence_df['Volumename'].isin(volume_list)]
presence_df_out.to_csv("./subset_200/train_vqa_presence.csv", index=False)


size_path = "./subset_2000/train_vqa_size.csv"
size_df = pd.read_csv(size_path)
size_df_out = size_df[size_df['Volumename'].isin(volume_list)]
size_df_out.to_csv("./subset_200/train_vqa_size.csv", index=False)


report_path = "./subset_2000/train_region_report.csv"
report_df = pd.read_csv(report_path)
report_df_out = report_df[report_df['Volumename'].isin(volume_list)]
report_df_out.to_csv("./subset_200/train_region_report.csv", index=False)