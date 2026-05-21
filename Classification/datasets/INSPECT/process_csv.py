import pandas as pd
import numpy as np
import os

folder = "./"
csv_list = os.listdir(folder)
csv_list = [f for f in csv_list if f.endswith('.csv')]

for csv_file in csv_list:
    df = pd.read_csv(os.path.join(folder, csv_file))
    impression_id_list = df['impression_id'].tolist()
    label_list = df['pe_positive'].tolist()
    df_out = pd.DataFrame({'impression_id': impression_id_list, 'pe': label_list})
    df_out.to_csv(os.path.join(folder, csv_file), index=False)