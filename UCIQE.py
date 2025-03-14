########################entropy########################
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import scipy as cp
import math
import cv2
#image是增强后图像。在此处修改路径读取增强后图像。
image =cv2.imdecode(np.fromfile(r"C:\zzh\4x\UWCNN++\results\ruihua.png",dtype=np.uint8),-1)
 
#img2是增强前图像。在此处修改路径读取增强前图像。用于得到SSIM和MSE
#kernel=np.ones((5,5),np.uint8)
#img2=cv2.morphologyEx(image,cv2.MORPH_OPEN, kernel)
#img2=cv2.imdecode(np.fromfile(r"C:\zzh\大四下\UWCNN-master\TestCode\test_real\5.png",dtype=np.uint8),-1)
 
images=np.asarray(image)
plt.subplot(211)
plt.imshow(images)
plt.subplot(212)
images1=images.max(axis=2)
plt.imshow(images1,cmap='gray')
row,col=images1.shape[0],images1.shape[1] # 求图像的规格
images1_size=row*col # 图像像素点的总个数
H1=0
n=np.array([0 for i in range(256)]) # 产生一个 256 维数组
 
p=[]
for a in images1:
    for b in a:
        img_level=b # 获取图像的灰度级
        n[img_level]+=1 # 统计每个灰度级像素的点数
for k in range(256): # 循环
    v=n[k]/images1_size # 计算每一个像素点的概率
    p.append(v) # 为什么对数组赋值赋值不了
    if v!=0: # 如果像素点的概率不为零
        H1 += -v*math.log2(v) # 求熵值的公式
######################## entropy ########################
 
######################## SSIM,MSE ########################
#from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
#from skimage.metrics import mean_squared_error
#import numpy as np
 
#img1 = image
##img2 = cv2.imread(r'/home/kadinu/picture_processing/picture/5.png')
#psnr = peak_signal_noise_ratio(img1, img2)
#ssim = structural_similarity(img1, img2, multichannel=True)  # 对于多通道图像(RGB、HSV等)关键词multichannel要设置为True
#mse = mean_squared_error(img1, img2)
 
######################## SSIM,MSE ########################
 
######################## UCIQE，UIQM ########################
 
import cv2
import math
import numpy as np
 
hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)  # RGB转为HSV 
H, S, V = cv2.split(hsv)
delta = np.std(H) /180  #色度的标准差 
mu = np.mean(S) /255  #饱和度的平均值 
n, m = np.shape(V)
number = math.floor(n*m/100)  #所需像素的个数  
Maxsum, Minsum = 0, 0
V1, V2 = V /255, V/255
 
for i in range(1, number+1):
    Maxvalue = np.amax(np.amax(V1))
    x, y = np.where(V1 == Maxvalue)
    Maxsum = Maxsum + V1[x[0],y[0]]
    V1[x[0],y[0]] = 0
 
top = Maxsum/number
 
for i in range(1, number+1):
    Minvalue = np.amin(np.amin(V2))
    X, Y = np.where(V2 == Minvalue)
    Minsum = Minsum + V2[X[0],Y[0]]
    V2[X[0],Y[0]] = 1
 
bottom = Minsum/number
 
conl = top-bottom
 ###对比度 
uciqe = 0.4680*delta + 0.2745*conl + 0.2575*mu
 
 ###############
def uicm(img):
    b, r, g = cv2.split(img)
    RG = r - g
    YB = (r + g)/2 - b
    m, n, o = np.shape(img)  #img为三维 rbg为二维
    K = m*n
    alpha_L = 0.1
    alpha_R = 0.1  ##参数α 可调
    T_alpha_L = math.ceil(alpha_L*K)  #向上取整
    T_alpha_R = math.floor(alpha_R*K) #向下取整
 
    RG_list = RG.flatten()
    RG_list = sorted(RG_list)
    sum_RG = 0
    for i in range(T_alpha_L+1, K-T_alpha_R ):
        sum_RG = sum_RG + RG_list[i]
    U_RG = sum_RG/(K - T_alpha_R - T_alpha_L)
    squ_RG = 0
    for i in range(K):
        squ_RG = squ_RG + np.square(RG_list[i] - U_RG)
    sigma2_RG = squ_RG/K
 
    YB_list = YB.flatten()
    YB_list = sorted(YB_list)
    sum_YB = 0
    for i in range(T_alpha_L+1, K-T_alpha_R ):
        sum_YB = sum_YB + YB_list[i]
    U_YB = sum_YB/(K - T_alpha_R - T_alpha_L)
    squ_YB = 0
    for i in range(K):
        squ_YB = squ_YB + np.square(YB_list[i] - U_YB)
    sigma2_YB = squ_YB/K
 
    Uicm = -0.0268*np.sqrt(np.square(U_RG) + np.square(U_YB)) + 0.1586*np.sqrt(sigma2_RG + sigma2_YB)
    return Uicm
 
def EME(rbg, L):
    m, n = np.shape(rbg)  #横向为n列 纵向为m行
    number_m = math.floor(m/L)
    number_n = math.floor(n/L)
    #A1 = np.zeros((L, L))
    m1 = 0
    E = 0
    for i in range(number_m):
        n1 = 0
        for t in range(number_n):
            A1 = rbg[m1:m1+L, n1:n1+L]
            rbg_min = np.amin(np.amin(A1))
            rbg_max = np.amax(np.amax(A1))
 
            if rbg_min > 0 :
                rbg_ratio = rbg_max/rbg_min
            else :
                rbg_ratio = rbg_max  ###
            E = E + np.log(rbg_ratio + 1e-5)
 
            n1 = n1 + L
        m1 = m1 + L
    E_sum = 2*E/(number_m*number_n)
    return E_sum
 
def UICONM(rbg, L):  #wrong
    m, n, o = np.shape(rbg)  #横向为n列 纵向为m行
    number_m = math.floor(m/L)
    number_n = math.floor(n/L)
    A1 = np.zeros((L, L)) #全0矩阵
    m1 = 0
    logAMEE = 0
    for i in range(number_m):
        n1 = 0
        for t in range(number_n):
            A1 = rbg[m1:m1+L, n1:n1+L]
            rbg_min = int(np.amin(np.amin(A1)))
            rbg_max = int(np.amax(np.amax(A1)))
            plip_add = rbg_max+rbg_min-rbg_max*rbg_min/1026
            if 1026-rbg_min > 0:
                plip_del = 1026*(rbg_max-rbg_min)/(1026-rbg_min)
                if plip_del > 0 and plip_add > 0:
                    local_a = plip_del/plip_add
                    local_b = math.log(plip_del/plip_add)
                    phi = local_a * local_b
                    logAMEE = logAMEE + phi
            n1 = n1 + L
        m1 = m1 + L
    logAMEE = 1026-1026*((1-logAMEE/1026)**(1/(number_n*number_m)))
    return logAMEE
 
if __name__ == '__main__':
    img = image
    r, b, g = cv2.split(img)
 
    Uicm = uicm(img)
 
    EME_r = EME(r, 8)
    EME_b = EME(b, 8)
    EME_g = EME(g, 8)
    Uism = 0.299*EME_r + 0.144*EME_b + 0.557*EME_g
 
    Uiconm = UICONM(img, 8)
 
    uiqm = 0.0282*Uicm + 0.2953*Uism + 0.6765*Uiconm    
    
    ######################## UCIQE，UIQM ########################
#print('Entropy：{}，PSNR：{}，SSIM：{}，MSE：{}，UCIQE：{}，UIQM：{}'.format(H1,psnr, ssim, mse,uciqe,uiqm))
print('Entropy：{}，UCIQE：{}，UIQM：{}'.format(H1,uciqe,uiqm))

