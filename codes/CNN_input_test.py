
import numpy as np
from keras.datasets import mnist
from keras.models import load_model
from keras.utils import to_categorical

model = load_model('mnist_cnn_model.h5')
model.load_weights('mnist_cnn_weights.h5')
(x_train, y_train), (x_test, y_test) = mnist.load_data()

input_test = np.load('input_test.npz')
input_test_set = np.asarray(input_test['input_test']) #test는 numpy형식으로 input을 줘야하므로, npz를 numpy형식으로 변환
#print(input_test_set)


num_classes = 10  # 0부터 9까지 총 10개의 classes


#직접 넣은 input으로 predict 하는 코드.
predict = model.predict(input_test_set)
# print(predict)
print('Result : ', np.argmax(predict, axis=1))
