import numpy as np
import pandas as pd
import os
import sys

#python train_test_split.py Path_to_data_csv test_split_percentage
# ex:: python train_test_split.py dataset/data.csv 0.1


csv_file_path = sys.argv[1]
test_split_percentage= sys.argv[2]

df = pd.read_csv(csv_file_path)
number_of_test_samples_per_class= int((len(df) *0.1)/len(df["label"].unique()))
Test_dataframe = pd.DataFrame()
folder_path= csv_file_path[:-len(os.path.basename(csv_file_path))]
for class_label in df['label'].unique():
    df_class = df.loc[(df['label']==class_label)]
    df_class = df_class.sample(frac=1) #shuffle rows
    df_class = df_class[:number_of_test_samples_per_class]
    df = df[~df.index.isin(df_class.index)]
    Test_dataframe =Test_dataframe.append(df_class)

Test_dataframe.to_csv(folder_path+'Test_data.csv', index=False)
df.to_csv(folder_path+'Train_data.csv', index=False)