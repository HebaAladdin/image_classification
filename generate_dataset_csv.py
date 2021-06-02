import numpy as np
import pandas as pd
from glob import glob
import os
import sys

# python generate_dataset_csv.py Path_to_dataset
# ex python generate_dataset_csv.py dataset/

csv_dic={"image_path":[], "label":[]}

dataset_folder= sys.argv[1]
images_paths = glob(dataset_folder+'/*/*[.png , .jpg]')

for images_path in images_paths:
    csv_dic["image_path"].append(images_path)
    csv_dic["label"].append(images_path.split('/')[-2])

df=pd.DataFrame(csv_dic)
df.to_csv(os.path.join(dataset_folder,'data.csv'),index=False)