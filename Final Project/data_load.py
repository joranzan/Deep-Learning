import cv2
import os
path_name = 'extra_food/extra_food'
path_name2 = 'extra_food/no_extra_food'
files = os.listdir(path_name)
files2 = os.listdir(path_name2)

for i in files:
    name = i[0:-5]
    img_path = cv2.imread('{}/{}.jpeg'.format(path_name,name), cv2.IMREAD_COLOR)

    dst = cv2.resize(img_path, dsize=(150, 150), interpolation=cv2.INTER_AREA)
    cv2.imwrite('data/extra_food/test{}.jpg'.format(name), dst)

for i in files2:
    name = i[0:-5]
    img_path2 = cv2.imread('{}/{}.jpeg'.format(path_name2,name), cv2.IMREAD_COLOR)

    dst2 = cv2.resize(img_path2, dsize=(150, 150), interpolation=cv2.INTER_AREA)
    cv2.imwrite('data/no_extra_food/test{}.jpg'.format(name), dst2)

