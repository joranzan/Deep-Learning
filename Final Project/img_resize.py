import cv2
import os



# train trans
path_name = 'train/exist'
files = os.listdir(path_name)


height , width = 256, 256

for i in files :
    img_path = cv2.imread('{}/{}'.format(path_name, i) , cv2.IMREAD_COLOR)

    dst = cv2.resize(img_path, dsize=(height, width), interpolation=cv2.INTER_AREA)
    cv2.imwrite('./trans/{}/{}.jpg'.format(path_name,i), dst)



path_name = 'train/nonexist'
files = os.listdir(path_name)

for i in files :
    img_path = cv2.imread('{}/{}'.format(path_name, i) , cv2.IMREAD_COLOR)

    dst = cv2.resize(img_path, dsize=(height, width), interpolation=cv2.INTER_AREA)
    cv2.imwrite('./trans/{}/{}.jpg'.format(path_name,i), dst)



# test trans

path_name = 'test/exist'
files = os.listdir(path_name)
for i in files :
    img_path = cv2.imread('{}/{}'.format(path_name, i) , cv2.IMREAD_COLOR)

    dst = cv2.resize(img_path, dsize=(height, width), interpolation=cv2.INTER_AREA)
    cv2.imwrite('./trans/{}/{}.jpg'.format(path_name,i), dst)


path_name = 'test/nonexist'
files = os.listdir(path_name)
for i in files :
    img_path = cv2.imread('{}/{}'.format(path_name, i) , cv2.IMREAD_COLOR)

    dst = cv2.resize(img_path, dsize=(height, width), interpolation=cv2.INTER_AREA)
    cv2.imwrite('./trans/{}/{}.jpg'.format(path_name,i), dst)