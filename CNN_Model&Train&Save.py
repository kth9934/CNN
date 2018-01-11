from keras.datasets import mnist
from keras.models import load_model
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.optimizers import adam


# hyper parameters => batch size, epoch, num_classes

batch_size = 100  # 한번에 훈련시킬 인풋데이터 개수
epoch = 1         # 모든 인풋 데이터셋을 한번씩 다 쓰면 1 epoch
num_classes = 10  # 0부터 9까지 총 10개의 classes

# mnist 데이터를 불러옵니다. train data와 test data로 자동 split됩니다
# 여기까지 해보고 x_train 과 x_test 의 shape 을 확인해봅시다

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print('shape')
# print(x_train.shape)
# print(y_train.shape)

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
# print(x_test[1,:,:])

# 데이터 값을 uint8 에서 float32로 바꾸고
# 0~255 사이의 값으로 이루어진 데이터를 0~1 사이의 값으로 노멀라이즈 해줍니다.

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

print(x_test[1])
print(y_test[1])

x_train /= 255
x_test /= 255

# 0~9 사이로 되어있는 값들을 [1, 0, 0, 0, 0, 0, 0, 0, 0] ~ [0, 0, 0, 0, 0, 0, 0, 0, 1]
# 사이의 값으로 변환해줍니다. 이런 변환을 원-핫 인코딩 이라고 합니다.

print(y_train[:10])
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
print(y_train[:10])

# 이제 CNN 레이어를 쌓아봅시다

model = Sequential()   #전체를 쌓아넣을 큰 틀을 만듬. 보통 순서는 convolution(with relu) - pooling 순으로 함.
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
#model.add => 틀에 레이어를 집어넣자. 32는 filter의 개수. 필터 초기화는 케라스가 알아서 렌덤으로 해줌.
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))  #stride가 행과 열으로 각각 2임.
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
#Flatten()은 conv와 pooling을 마친 후에 Fully connected 들어가기 전에 input에 들어갈 matrix를 벡터로 만드는 것. (평평하게 폄)
model.add(Dense(128, activation='relu'))
#Dense는 Fully connected 에서 연산하는 forward propagation. 128은 다음 Layer의 유닛개수를 의미함.
model.add(Dense(num_classes, activation='softmax'))
#softmax는 num_classes = 10개로 나온 최종 output unit을 각각 확률로 표현해줌.)
#softmax 개념은 따로 찾아서 공부하기.


# model 을 compile 해줍니다
# optimizer 와 loss function을 선택해줍니다.
# 모델의 summary 를 출력해줍니다.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# train 시작 !

model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch, verbose=1, validation_data=(x_test, y_test))

# verbose = 1 이면 중간 처리과정을 다 보여주고, 0이면 안보여줌.

score = model.evaluate(x_test, y_test, verbose=0)


# Test data 에 대해 학습된 모델을 검증해 봅시다
# score[0] = loss 값
# score[1] = accuracy 값

loss = score[0]
accuracy = score[1]

print("Accuracy : ", accuracy, " loss : ", loss)



# 모델과 weight값을 저장
model_save = model.save('mnist_cnn_model.h5')
del model
model_weight = model.save_weights('mnist_cnn_weights.h5', overwrite=True)

