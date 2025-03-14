import os
import cv2 as cv
 
width = 620
height = 460
img = cv.imread(r"C:\zzh\4x\UWCNN++\test_images\770.png")
#例如cv.imread("test/1.jpg")
 
img = cv.resize(img,(width,height))
# 默认使用双线性插值法

out_file_name = "10103out"
save_path = r"C:\zzh\4x\UWCNN++\test_images"
save_path_file = os.path.join(save_path,out_file_name+".png")
cv.imwrite(save_path_file,img)
