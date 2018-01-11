
import numpy as np
from keras.datasets import mnist
from keras.models import load_model
from keras.utils import to_categorical

model = load_model('mnist_cnn_model.h5')
model.load_weights('mnist_cnn_weights.h5')
(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_test = x_test.astype('float32')
x_test /= 255


num_classes = 10  # 0부터 9까지 총 10개의 classes

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

predict = model.predict(x_test[1:10])
#print(predict)
print('Answers :', np.argmax(predict, axis=1))
print('Results :', np.argmax(y_test[1:10], axis=1))