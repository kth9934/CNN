from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential, load_model
from keras.utils import to_categorical
import numpy as np
from keras.losses import categorical_crossentropy
from keras.optimizers import adam
from PIL import Image, ImageOps
import matplotlib.pylab as plt

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

# hyper parameters => batch size, epoch, num_classes

batch_size = 100
epoch = 1
num_classes = 10

# mnist 데이터를 불러옵니다. train data와 test data로 자동 split됩니다
# 여기까지 해보고 x_train 과 x_test 의 shape 을 확인해봅시다

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print('shape')
print(x_train.shape)
print(y_train.shape)
# 이미지 확인
# plt.imshow(x_train[0], cmap=plt.get_camp('gray'))
# plt.show()


# train 데이터의 shape은 (60000, 28, 28) 입니다.
# 가장 앞에 사진의 개수 첫 번째가 x축 길이 두 번째 가 y축의 길이를 나타냅니다
# 보통 사진은 (number, x_img size, y_imag size, channels)의 형태로 나타내게 되는데
# 가장 끝에는 칼라사진인 경우 RGB 세 가지의 값이 들어가므로 3 이 들어가게 됩니다
# 하지만 mnist는 흑백 사진으로 RGB가 아니기 때문에 1을 넣어줍니다.
# 따라서 (60000, 28, 28) - > (60000, 28, 28, 1) 이렇게 변환해줍니다.

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# 데이터 값을 uint8 에서 float32로 바꾸고
# 0~255 사이의 값으로 이루어진 데이터를 0~1 사이의 값으로 노멀라이즈 해줍니다.

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

# 0~9 사이로 되어있는 값들을 [1, 0, 0, 0, 0, 0, 0, 0, 0] ~ [0, 0, 0, 0, 0, 0, 0, 0, 1]
# 사이의 값으로 변환해줍니다. 이런 변환을 원-핫 인코딩 이라고 합니다.

print(y_train[:10])
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
print(y_train[:10])
# 이제 CNN 레이어를 쌓아봅시다

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# model 을 compile 해줍니다
# optimizer 와 loss function을 선택해줍니다
# 모델의 summary 를 출력해줍니다
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()

model.load_weights('weight.hdf5')
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# im = x_test[1]
im = np.asarray(ImageOps.invert(Image.open('./img/5_resized.jpg')))

im = rgb2gray(im)
# plt.imshow(im, cmap=plt.get_cmap('gray'))
# plt.show()
im = im.reshape(1, 28, 28, 1)
im = im.astype('float32')
im /= 255
print(im)



predict = model.predict(im)
print(predict)
print(np.argmax(predict, axis=1))