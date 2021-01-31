import pandas as pd
from create_tfrecord import tfrecord
import os
import matplotlib.pyplot as plt

def label_transfer(dataset_dir):

    #convert label from 5 classes to 2 classes

    train_dir = os.path.join(dataset_dir, r"labels\train.csv")
    test_dir = os.path.join(dataset_dir, r"labels\test.csv")

    df_train = pd.read_csv(train_dir)
    df_test = pd.read_csv(test_dir)

    for index, row in df_train.iterrows():
        if row['Retinopathy grade'] <= 1:
            df_train.loc[index, 'Retinopathy grade'] = 0
        else:
            df_train.loc[index, 'Retinopathy grade'] = 1

    for index, row in df_test.iterrows():
        if row['Retinopathy grade'] <= 1:
            df_test.loc[index, 'Retinopathy grade'] = 0
        else:
            df_test.loc[index, 'Retinopathy grade'] = 1
    df_train.to_csv(os.path.join(dataset_dir, r"train_binary.csv"), index=False)
    df_test.to_csv(os.path.join(dataset_dir, r"test_binary.csv"), index=False)
    return df_train, df_test

def EDA(data):

    #Visualize dataset distribution

    data = data['Retinopathy grade']
    data_value = data.value_counts()
    plt.bar(data_value.index, data_value)
    plt.xticks(data_value.index, data_value.index.values)
    plt.xlabel("labels")
    plt.ylabel("Frequency")
    plt.title('Distribution of diabetic retinopathy in test dataset')
    plt.show()

# change dataset_dir to your own dir
dataset_dir = "E:\idrid\IDRID_dataset"
# convert 5 classification to 2 classification
train_dataset, test_dataset = label_transfer(dataset_dir)
# create tfrecord files
tfrecord(train_dataset, test_dataset, dataset_dir)
# visualized data distribution
EDA(train_dataset)
EDA(test_dataset)
