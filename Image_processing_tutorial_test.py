#image를 처리하고 읽기 위해서 pillow, numpy를 깔아야한다.

from PIL import Image
import numpy as np
from tempfile import TemporaryFile
test_dataset = TemporaryFile()
# import matplotlib.pyplot as plt

#dictionary를 만든다. dictionary는 Key와 value로 구성.
#Key 안에 value들이 있다.
image_dict = { }
image_pix_dict = { }

#i가 0부터 10개. 즉 0부터 9까지.
#i가 돌면서 10개의 key들을 생성하고, 각 key에 value들을 i.jpg를 여는 것으로 할당한다.
for i in range(10):
    image_dict["image_{}".format(i)] = Image.open('{}.jpg'.format(i))
    #print(image_dict["image_0"])
# 저장된 key와 value들을 보여준다.
# print(image_dict.keys())
# print(image_dict.values())

#각 Key에 할당된 value들(i.jpg들)을 모두 28 by 28로 resize한다.
    image_dict["image_{}".format(i)] = image_dict["image_{}".format(i)].resize((28,28))

    image_dict["image_{}".format(i)].save('{}_resized.jpg'.format(i))
    image_dict["image_{}".format(i)] = image_dict["image_{}".format(i)].convert("L") #channel을 단일채널로 변환
    image_pix_dict["pix_{}".format(i)] = 255 - np.array(image_dict["image_{}".format(i)])  # values를 수정가능한 numpy 행렬로 변환
    image_pix_dict["pix_{}".format(i)] = image_pix_dict["pix_{}".format(i)].reshape(28, 28, 1)  # reshape


# # print(image_pix_dict["pix_0"][0,0,0])
#
# # 인풋의 회색부분을 모두 흰색으로 만들어주기 위한 작업
# for i in range(10):
#     for j in range(28):
#         for k in range(28):
#             if image_pix_dict["pix_{}".format(i)][j,k,0] > 150:
#                 image_pix_dict["pix_{}".format(i)][j,k,0] = image_pix_dict["pix_{}".format(i)][j,k,0]
#             else:
#                 image_pix_dict["pix_{}".format(i)][j, k, 0] = 0
#     image_pix_dict["pix_{}".format(i)] = image_pix_dict["pix_{}".format(i)].astype('float32')
#     image_pix_dict["pix_{}".format(i)] /= 255
#
#
# input_test = np.expand_dims(image_pix_dict["pix_0"],axis=0)
# for i in range(10-1):
#     image_pix_dict["pix_{}".format(i+1)] = np.expand_dims(image_pix_dict["pix_{}".format(i+1)], axis=0)
#     input_test = np.concatenate([input_test, image_pix_dict["pix_{}".format(i+1)]], axis=0)
# # 각각의 헹렬(0부터 9까지)를 하나의 행렬에 집어 넣는다.
# # 우선 input_test를, 숫자 0을 나타내는 행렬의 차원에서 맨 앞 쪽에 새로운 차원을 추가한 것으로 설정한다. (28,28,1) -> (1, 28, 28, 1)
# # 그 후 for문을 이용해서 1부터 9까지 모두 맨 앞쪽에 새로운 차원을 추가 해주고,
# # input_test를 새로운 input_test ( 새로운 행렬이 추가된 input_test)로 업데이트 해준다.
# # 아래와 같이 노가다로 해도 된다.
#
# # input_test = np.concatenate((image_pix_dict["pix_{}".format(0)],
# # image_pix_dict["pix_{}".format(1)],
# # image_pix_dict["pix_{}".format(2)],
# # image_pix_dict["pix_{}".format(3)],
# # image_pix_dict["pix_{}".format(4)],
# # image_pix_dict["pix_{}".format(5)],
# # image_pix_dict["pix_{}".format(6)],
# # image_pix_dict["pix_{}".format(7)],
# # image_pix_dict["pix_{}".format(8)],
# # image_pix_dict["pix_{}".format(9)]), axis=0 )
#
# # print(input_test.shape) #행렬들이 잘 들어갔는지 테스트 출력
#
# np.savez('input_test.npz', input_test=input_test) #테스트 셋을 저장한다.
#
# # data = np.load('input_test.npz') #테스트 셋이 잘 저장되었는지 확인하기 위해 다시 불러온 후 출력.
# # print(data['input_test'])

