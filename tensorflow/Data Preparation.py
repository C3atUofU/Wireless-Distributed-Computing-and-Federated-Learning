import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
from PIL import Image
import requests
from sklearn.model_selection import train_test_split
import warnings
from io import BytesIO

warnings.filterwarnings('ignore')

IMAGE_SIZE = (224, 224)

training_data = pd.read_csv('C:/Users/dddoo/Desktop/Project/train.csv')

response = requests.get(training_data['url'][0])
ranges = ['1-5', '5-10', '10-50', '50-100', '100-500', '500-1000', '>=1000']
pics_count = pd.DataFrame(training_data.landmark_id.value_counts())

# Create a column where to group the number of images in each class
pics_count['range'] = np.where(pics_count['landmark_id'] >= 1, '1-5', 0)
pics_count['range'] = np.where(pics_count['landmark_id'] >= 5, '5-10', pics_count['range'])
pics_count['range'] = np.where(pics_count['landmark_id'] >= 10, '10-50', pics_count['range'])
pics_count['range'] = np.where(pics_count['landmark_id'] >= 50, '50-100', pics_count['range'])
pics_count['range'] = np.where(pics_count['landmark_id'] >= 100, '100-500', pics_count['range'])
pics_count['range'] = np.where(pics_count['landmark_id'] >= 500, '500-1000', pics_count['range'])
pics_count['range'] = np.where(pics_count['landmark_id'] >= 1000, '>=1000', pics_count['range'])

pics_count['range'].value_counts().loc[ranges].plot(kind='bar', width=0.7)
plt.title('Distribution of images per class')
plt.xlabel('Images')
plt.ylabel('Classes')
plt.show()

# Select 100 classes containing between 10 and 50 images
sample_list = pics_count[pics_count['range']=='10-50'][0:100]
data_sample = training_data[training_data['landmark_id'].isin(sample_list.index)]
# Reorder sub-sample classes from 0 to 9
old_classes = list(set(data_sample['landmark_id']))
new_classes = list(range(100))
data_sample['landmark_id'] = data_sample['landmark_id'].replace(to_replace=old_classes, value=new_classes)


def dataframe_split(dataframe, validation_size, test_size):
    dataframe, test_df = train_test_split(dataframe, test_size=test_size)

    training_df = pd.DataFrame(columns=['id', 'url', 'landmark_id'])
    validation_df = pd.DataFrame(columns=['id', 'url', 'landmark_id'])
    random.seed(7)

    # Split the dataset class by class
    # 'cc' stands for current class
    for cc_landmark_id in set(dataframe['landmark_id']):
        cc_data = dataframe[(dataframe.landmark_id == cc_landmark_id)]
        i = 0
        cc_images = []
        while i < len(cc_data.id):
            cc_images.append(cc_data.id.iloc[i])  # List of all the images for the Current Class
            i += 1

        # Randomly pick a sample of images for test and get the remaining images for training
        cc_validation_images = random.sample(cc_images, int(validation_size * len(cc_images)))
        cc_training_images = list(set(cc_images) - set(cc_validation_images))

        # Dataset creation from the Image id's
        cc_validation_df = dataframe[dataframe.id.isin(cc_validation_images)]
        cc_training_df = dataframe[dataframe.id.isin(cc_training_images)]

        # Add to the whole datasets
        validation_df = validation_df.append(cc_validation_df)
        training_df = training_df.append(cc_training_df)
    return training_df, validation_df, test_df

training_data, validation_data, test_data = dataframe_split(data_sample, validation_size=0.2, test_size=0.05)

# Reset indices
training_data.reset_index(inplace=True)
validation_data.reset_index(inplace=True)
test_data.reset_index(inplace=True)


def images_download_store(dataset, data_folder):
    landmarks_ids = pd.DataFrame(dataset.landmark_id.value_counts())
    landmarks_ids.reset_index(inplace=True)
    landmarks_ids.columns = ['landmark_id', 'count']

    # Create Landmark's folders
    i = 0
    while i < len(dataset):
        landmark = str(dataset.landmark_id.iloc[i])
        if not os.path.exists('Data/' + data_folder + '/' + landmark):
            os.makedirs('Data/' + data_folder + '/' + landmark)
        i += 1

    # Download Images
    i = 0
    for url in dataset['url']:
        if os.path.exists('Data/' + data_folder + '/' + str(dataset['landmark_id'].iloc[i]) + '/' + str(
                dataset['id'].iloc[i]) + '.jpg'):
            i += 1
            continue
        try:
            response = requests.get(url, stream=True)
            image = Image.open(BytesIO(response.content))
            image = image.resize(IMAGE_SIZE, Image.ANTIALIAS)
            image.save('Data/' + data_folder + '/' + str(dataset['landmark_id'].iloc[i]) + '/' + str(
                dataset['id'].iloc[i]) + '.jpg')
            del response
        except:
            pass
        i += 1
        if (i % 100 == 0):
            print(str(i) + ' images downloaded')
    print('Images downloaded')

# images_download_store(training_data,'Training_Data')
# print('Training Images Downloaded')
# images_download_store(validation_data,'Validation_Data')
# print('Validation Images Downloaded')
# images_download_store(test_data,'Test_Data')
# print('Test Images Downloaded')

train_data_dir = 'Data/Training_Data'
validation_data_dir = 'Data/Validation_Data'
test_data_dir= 'Data/Test_Data'
NB_CLASSES = 11

train_labels_file = 'Labels/training_labels_1.npy'
validation_labels_file = 'Labels/validation_labels_1.npy'
test_labels_file = 'Labels/test_labels_1.npy'

def get_labels(dir):
    i = 0
    labels = []
    while i < NB_CLASSES:
        for root, dirs, files in os.walk(dir + '/' + str(i)):  # Loop through folders
            for pic in files:
                try:
                    Image.open(dir + '/' + str(i) + '/' + pic)
                    labels.append(str(i))
                except:
                    os.remove(dir + '/' + str(i) + '/' + pic)  # Remove broken files
        i += 1
    print(dir + ' : ' + str(len(labels)) + ' Images')
    return labels

train_labels = get_labels(train_data_dir)
validation_labels = get_labels(validation_data_dir)
test_labels = get_labels(test_data_dir)

with open(train_labels_file, 'wb') as file:
    np.save(file, train_labels)

with open(validation_labels_file, 'wb') as file:
    np.save(file, validation_labels)

with open(test_labels_file, 'wb') as file:
    np.save(file, test_labels)