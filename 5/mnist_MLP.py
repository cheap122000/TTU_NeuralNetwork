from keras.datasets import mnist
(train_feature, train_label), (test_feature, test_label) = mnist.load_data()

print(len(train_feature), len(train_label))
print(train_feature.shape, train_label.shape)

import matplotlib.pyplot as plt

def show_image(image):
    fig=plt.gcf()
    fig.set_size_inches(2, 2) # 圖片大小
    plt.imshow(image, cmap='binary') # 黑白
    plt.show()

show_image(train_feature[0])
print(train_label[0])

def show_images_labels_predictions(images, labels, predictions, start_id, num=10):
    plt.gcf().set_size_inches(12, 14)
    if num > 25: num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, i+1)
        ax.imshow(images[start_id], cmap='binary')
        
        if len(predictions) > 0:
            title = 'ai = ' + str(predictions[start_id])
            title += (' (o)' if predictions[start_id] == labels[start_id] else ' (x)')
            title += '\nlabel = ' + str(labels[start_id])
        else:
            title = 'label = ' + str(labels[start_id])
            
        ax.set_title(title, fontsize = 12)
        ax.set_xticks([])
        ax.set_yticks([])
        start_id += 1
    plt.show()

show_images_labels_predictions(train_feature, train_label, [], 0, 10)

train_feature_vector = train_feature.reshape(len(train_feature), 784).astype('float32')
test_feature_vector = test_feature.reshape(len(test_feature), 784).astype('float32')

print(train_feature_vector.shape, test_feature_vector.shape)

print(train_feature_vector[0])
print(test_feature_vector[0])

train_feature_normalize = train_feature_vector/255
test_feature_normalize = test_feature_vector/255

print(train_label[0:5])

from keras.utils import np_utils
train_label_onehot = np_utils.to_categorical(train_label)
test_label_onehot = np_utils.to_categorical(test_label)

print(train_label_onehot[0:5])

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(units=256,
                input_dim=784,
                kernel_initializer='normal',
                activation='relu'))
model.add(Dense(units=10,
                kernel_initializer='normal',
                activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

train_history = model.fit(x=train_feature_normalize,
                          y=train_label_onehot,
                          validation_split=0.2,
                          epochs=10,
                          batch_size=200,
                          verbose=2)

scores = model.evaluate(test_feature_normalize, test_label_onehot)
print('\n準確率 = ', scores[1])

prediction = model.predict_classes(test_feature_normalize)

show_images_labels_predictions(test_feature, test_label, prediction, 0)

model.save('Mnist_mlp_model.h5')
print('Mnist_mlp_model.h5 儲存完畢！')
del model

from keras.models import load_model
model = load_model('Mnist_mlp_model.h5')
prediction = model.predict_classes(test_feature_normalize)
show_images_labels_predictions(test_feature, test_label, prediction, 50, num=30)