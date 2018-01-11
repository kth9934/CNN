from keras.datasets import mnist
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



# train 데이터의 shape은 (60000, 28, 28) 입니다.
# 가장 앞에 사진의 개수 첫 번째가 x축 길이 두 번째 가 y축의 길이를 나타냅니다
# 보통 사진은 (number, x_img size, y_imag size, channels)의 형태로 나타내게 되는데
# 가장 끝에는 칼라사진인 경우 RGB 세 가지의 값이 들어가므로 3 이 들어가게 됩니다
# 하지만 mnist는 흑백 사진으로 RGB가 아니기 때문에 1을 넣어줍니다.
# 따라서 (60000, 28, 28) - > (60000, 28, 28, 1) 이렇게 변환해줍니다.

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

reshpe_x_train = x_train.reshape(60000, 28, 28, 1)


print(x_train[1])
