import math
import numpy as np
import tensorflow as tf
import keras
from keras import applications, Model
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Flatten, Dense, BatchNormalization
from keras.utils.np_utils import to_categorical
import warnings
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()
warnings.simplefilter('ignore')

train_data_dir = 'Data/Training_Data'
validation_data_dir = 'Data/Validation_Data'
test_data_dir= 'Data/Test_Data'

training_features_file = 'Features/training_features_VGG16.npy'
validation_features_file = 'Features/validation_features_VGG16.npy'
top_weights_file = 'Weights/weights_VGG16.h5'
model_file = 'Models/model_vgg16.h5'

train_labels_file = 'Labels/training_labels_1.npy'
validation_labels_file = 'Labels/validation_labels_1.npy'
test_labels_file = 'Labels/test_labels_1.npy'

img_width, img_height = 224, 224
NB_CLASSES = 11

train_labels = np.load(open(train_labels_file, 'rb'))
validation_labels = np.load(open(validation_labels_file, 'rb'))
test_labels = np.load(open(test_labels_file, 'rb'))

print('Training Data : ' + str(len(train_labels)) + ' Images')
print('Validation Data : ' + str(len(validation_labels)) + ' Images')
print('Test Data : ' + str(len(test_labels)) + ' Images')


def images_to_feature_vectors(model, directory, batch_size, steps):
    datagen = ImageDataGenerator()

    generator = datagen.flow_from_directory(
        directory,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)  # Keep the data in the same order

    features = model.predict_generator(generator, steps, verbose=1)

    return features


# Batch size has to be a multiple of the number of images  to keep our vectors consistents
training_batch_size = 1 # batch size for feature pre-training
validation_batch_size = 1 # batch size for feature pre-training

base_model = applications.VGG16(include_top=False, weights='imagenet', input_shape=(img_width,img_height,3)) #VGG16 trained on imagenet
training_features = images_to_feature_vectors(base_model, train_data_dir, training_batch_size, len(train_labels) // training_batch_size)
validation_features = images_to_feature_vectors(base_model, validation_data_dir, validation_batch_size, len(validation_labels) // validation_batch_size)

with open(training_features_file, 'wb') as file:
    np.save(file, training_features, allow_pickle=False)
with open(validation_features_file, 'wb') as file:
    np.save(file, validation_features, allow_pickle=False)

train_data = np.load(open(training_features_file, 'rb'))
validation_data = np.load(open(validation_features_file, 'rb'))

# def create_model(lr, decay, nn1, nn2, nn3, input_shape, output_shape):
#     '''This is a model generating function so that we can search over neural net
#     parameters and architecture'''
#
#     opt = keras.optimizers.Adam(lr=lr, decay=decay)
#
#     model = Sequential()
#
#     model.add(Flatten())
#     model.add(Dense(nn1, activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Dense(nn2, activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Dense(nn3, activation='relu'))
#     model.add(BatchNormalization())
#
#     model.add(Dense(output_shape, activation='softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'], )
#     return model
#
# model = KerasClassifier(build_fn=create_model, epochs=16, verbose=1)
#
# # Learning rate values
# lr=[1e-2, 1e-3, 1e-4]
# decay=[1e-6,1e-9,0]
#
# # Number of neurons per layer
# nn1=[4096,2048,1024]
# nn2=[2048,1024,512]
# nn3=[1000,500,200]
#
# batch_size=[2048,1024,512]
#
# train_data = np.load(open(training_features_file, 'rb'))
# # train_data = train_data.reshape(train_data.shape[0],-1)
# validation_data = np.load(open(validation_features_file, 'rb'))
# # validation_data = validation_data.reshape(validation_data.shape[0],-1)
#
# train_labels_onehot = to_categorical(train_labels, NB_CLASSES)  # One Hot Encoder
# validation_labels_onehot = to_categorical(validation_labels, NB_CLASSES)  # One Hot Encoder
#
# # dictionary summary
# param_grid = dict(
#                     lr=lr, decay=decay, nn1=nn1, nn2=nn2, nn3=nn3,
#                     batch_size=batch_size,
#                     input_shape=train_data.shape[1:], output_shape = (NB_CLASSES,)
#                  )
#
#
# grid = RandomizedSearchCV(estimator=model, cv=KFold(3), param_distributions=param_grid,
#                           verbose=20,  n_iter=10, n_jobs=1)
#
# grid_result = grid.fit(train_data, train_labels_onehot)
# cv_results_df = pd.DataFrame(grid_result.cv_results_)
# cv_results_df.to_csv('gridsearch_VGG16.csv')
# print(cv_results_df)
# print(grid_result.best_params_)

# Parameters after RandomizedSearchCV
nn1 = 5120; nn2 = 1024; nn3 = 200;
lr = 0.001; decay=1e-9
batch_size = 2048

# Regularization Parameters
dropout = 0.5
l1 = 0.0001
l2 = 0.0001

def model():
    opt = keras.optimizers.Adam(lr=lr)
    reg = keras.regularizers.l1_l2(l1=l1, l2=l2)

    model = Sequential()

    model.add(Flatten())
    model.add(Dense(nn1, activation='relu', kernel_regularizer=reg))
    model.add(Dropout(dropout))
    model.add(BatchNormalization())
    model.add(Dense(nn2, activation='relu', kernel_regularizer=reg))
    model.add(Dropout(dropout))
    model.add(BatchNormalization())
    model.add(Dense(nn3, activation='relu', kernel_regularizer=reg))
    model.add(Dropout(dropout))
    model.add(BatchNormalization())

    model.add(Dense(NB_CLASSES, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'], )
    return model

train_labels_onehot = to_categorical(train_labels, NB_CLASSES)  #One Hot Encoder
validation_labels_onehot = to_categorical(validation_labels, NB_CLASSES)  #One Hot Encoder

top = model()

history = top.fit(train_data, train_labels_onehot,
              epochs=80,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels_onehot))

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

top.save_weights(top_weights_file)

def top():
    opt = keras.optimizers.Adam(lr=lr)
    reg = keras.regularizers.l1_l2(l1=l1, l2=l2)

    base = applications.VGG16(include_top=False, weights='imagenet',
                              input_shape=(img_width, img_height, 3))  # VGG16 trained on Imagenet

    top = Sequential()

    top.add(Flatten(input_shape=train_data.shape[1:]))
    top.add(Dense(nn1, activation='relu', kernel_regularizer=reg))
    top.add(BatchNormalization())
    top.add(Dense(nn2, activation='relu', kernel_regularizer=reg))
    top.add(BatchNormalization())
    top.add(Dense(nn3, activation='relu', kernel_regularizer=reg))
    top.add(BatchNormalization())
    top.add(Dense(NB_CLASSES, activation='softmax'))
    top.load_weights(top_weights_file)
    top.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'], )

    model = Model(input=base.input, output=top(base.output))
    for layer in model.layers:
        layer.trainable = False
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'], )
    return model

complete_model = top()

datagen = ImageDataGenerator()

generator = datagen.flow_from_directory(
            test_data_dir,
            target_size=(img_width, img_height),
            batch_size=len(test_labels),
            class_mode=None,
            shuffle=False)

test_predictions = complete_model.predict_generator(generator, 1, verbose=1)
test_predictions = np.asarray(list(map(str,np.argmax(test_predictions,axis=1)))).reshape(-1, 1)
test_labels = np.asarray(test_labels).reshape(-1, 1)

test_acc, test_update_op = tf.compat.v1.metrics.accuracy(test_labels, test_predictions)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
sess.run(tf.compat.v1.local_variables_initializer())

test_accuracy = sess.run(test_update_op)
print('\nTest accuracy : ' + str(round(test_accuracy*100, 1)) + '%')