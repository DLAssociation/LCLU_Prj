# Unfinished!
import cv2
import numpy as np
import os

tiff_file = './try_img/2.tiff'
save_folder = './try_img_re/'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

tif_img = cv2.imread(tiff_file)
width, height, channel = tif_img.shape
# print height, width, channel : 6908 7300 3
threshold = 1000
overlap = 100

step = threshold - overlap
x_num = width / step + 1
y_num = height / step + 1
print(x_num, y_num)

N = 0
yj = 0

for xi in range(x_num):
    for yj in range(y_num):
        # print xi
        if yj <= y_num:
            print
            yj
            x = step * xi
        y = step * yj

        wi = min(width, x + threshold)
        hi = min(height, y + threshold)
        # print wi , hi

        if wi - x < 1000 and hi - y < 1000:
            im_block = tif_img[wi - 1000:wi, hi - 1000:hi]

        elif wi - x > 1000 and hi - y < 1000:
            im_block = tif_img[x:wi, hi - 1000:hi]

        elif wi - x < 1000 and hi - y > 1000:
            im_block = tif_img[wi - 1000:wi, y:hi]

        else:
            im_block = tif_img[x:wi, y:hi]

        cv2.imwrite(save_folder + 'try' + str(N) + '.jpg', im_block)
        N += 1