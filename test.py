import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt

thr_down_cnt = 70000 #min contour area to detect an object
thr_max_cnt = 500000 #max contour area to detect overlap object
max_contour_grab = 8
kernel = np.ones((3, 3), np.uint8)
min_dist = 100
offset = 15
ddepth = cv2.CV_16S
theshold = 50 # in set of 1 children, each arc have longer than circle
canny_min = 30
canny_max = 40

def find_mask(cnts,hierarchy,close):
    # clse decide offset for object, if True, take contour closer, False is more smooth, (survey again about accurate)
    hierarchy = hierarchy[0]
    area = np.array([cv2.contourArea(c) for c in cnts]).reshape(-1,1)
    mask1 = np.where((area > thr_down_cnt) & (area <thr_max_cnt))[0]
    last = np.array([sub[-1] for sub in hierarchy])
    if close:
        mask2 = np.where(last != 1)[0]
    else:
        mask2 = np.where(last == 1)[0]
    mask = np.intersect1d(mask1,mask2) #find index of contour with condition
    return mask

def gen_thresh(img,cnts,hierarchy,close=True): #input is image and raw contour find from threshold
    mask = find_mask(cnts,hierarchy,close)
    out = np.zeros_like(img).astype(np.uint8)
    for m in mask:
      cv2.drawContours(out,cnts,m,(255,255,255),-1)
    return out

def find_csd(ctr,img):
    new_r = (np.sqrt(2)/2)*ctr[2]
    inter_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(inter_img).astype(np.uint8)
    cv2.circle(mask,(ctr[0],ctr[1]),int(new_r)-offset,(255,255,255),-1)
    out = inter_img * mask
    h,w = img.shape[0:2]
    out = cv2.resize(out,(w//3,h//3),interpolation=cv2.INTER_AREA)
    return out

def grad(out):
    grad_x = cv2.Sobel(out,ddepth,dx=1,dy=0,ksize=3)
    grad_y = cv2.Sobel(out,ddepth,dx=0,dy=1,ksize=3)
    agx = cv2.convertScaleAbs(grad_x)
    agy = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(agx,0.5,agy,0.5,0)
    return grad
def clf(img,ctr):
    rs = np.full(ctr.shape[0],False)
    for i in range(ctr.shape[0]):
        out = find_csd(ctr[i],img)
        cny = cv2.Canny(out,canny_min,canny_max)
        cc,hie_over = cv2.findContours(cny,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        clen = np.array([cv2.arcLength(c,False) for c in cc]).reshape(-1,1)
        last = np.array([sub[-1] for sub in hie_over[0]])
        m1 = np.where(clen > theshold)[0]
        m2 = np.where(last == 1)[0] # only take each contour have parent is 1
        mm = np.intersect1d(m1,m2)
        if len(mm) != 0:
            rs[i] = True # down
    return rs

img_name = "Image00004.BMP" # /content/t_drive/MyDrive/Data_XLA_Paper/data_4_1_sen_25/Image00002.BMP
img = cv2.imread(img_name)

use = img.copy()
use = cv2.medianBlur(use,11)
use = cv2.pyrMeanShiftFiltering(use, 10,20)
gray = use[...,0]
thr = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,3)
cnts,hierarchy = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

rs = gen_thresh(use,cnts,hierarchy,False)
ctr = np.array([(555, 302, 284), (836, 398, 210), (512, 726, 199)]) # need input is center in radius 
# gradd = grad(out)

print(clf(use,ctr))


# plt.figure(figsize=(12,12))
# plt.subplot(1,2,1)
# plt.imshow(img[:,:,::-1])
# plt.title("color")
# plt.subplot(1,2,2)
# plt.imshow(rs, cmap='gray')
# plt.title("gray")
# plt.show()