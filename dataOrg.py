import os
import pandas as pd
import shutil

class_names = ['blackheads', 'dark_spot', 'nodules', 'papules', 'pustules', 'whiteheads']

data_dirs = ['datasetorg/train', 'datasetorg/test', 'datasetorg/valid']

def create_class_folders(base_dir, class_names):
    for class_name in class_names:
        class_folder = os.path.join(base_dir, class_name)
        os.makedirs(class_folder, exist_ok=True)

def sort_images(data_dir, class_names):
    csv_path = os.path.join(data_dir, 'classes.csv')  


    df = pd.read_csv(csv_path)
    print(df.head())

    create_class_folders('dataset', class_names)

    for index, row in df.iterrows():
        filename = row.iloc[0]
        labels = row.iloc[1:].values 

        for class_index, label in enumerate(labels):
            if label == 1:  
                class_folder = os.path.join('dataset', class_names[class_index])
                src = os.path.join(data_dir, filename)
                dst = os.path.join(class_folder, filename)
                shutil.copy(src, dst)

    # shutil.rmtree(data_dir)

for data_dir in data_dirs:
    sort_images(data_dir, class_names)
