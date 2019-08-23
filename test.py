import cv2
import tensorflow as tf
import AutoEncoder as model
import os
import imutils
import numpy as np

window_size=61
path='./test/'
files=os.listdir(path)
print(files)

for file in files:
    image=cv2.imread(path+file)
    shape=np.shape(image)
    mask=np.zeros((shape[0],shape[1]),dtype=np.uint8)
    dlmask=np.zeros((shape[0],shape[1]),dtype=np.uint8)

    with tf.Session() as sess:
        model.restore(sess)
        y=0
        while y<=shape[1]:
            x=0
            while x<=shape[0]:
                window=image[x:x+61,y:y+61]
                print(x,y)
                try: window=cv2.resize(window,(61,61),interpolation = cv2.INTER_CUBIC)
                except: break
                predict=sess.run(model.inference,feed_dict={model.x:[window]})
                if predict[0][1]>=0.8:
                    dlmask[x:x+61,y:y+61]=1
                    openkernel = np.ones((7,7),np.uint8)
                    closekernel = np.ones((7,7),np.uint8)

                    b,g,r = cv2.split(window)

                    blur = cv2.GaussianBlur(r,(7,7),0)
                    blur = cv2.GaussianBlur(blur,(7,7),0)
                    blur = cv2.GaussianBlur(blur,(7,7),0)
                    opening = cv2.morphologyEx(blur, cv2.MORPH_OPEN, openkernel)
                    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, closekernel)
                    patchret,patchmask = cv2.threshold(closing,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                    patchshape=np.shape(mask[x:x+61,y:y+61])
                    patchmask=cv2.resize(patchmask,(patchshape[1],patchshape[0]),interpolation=cv2.INTER_CUBIC)
                    mask[x:x+61,y:y+61]=patchmask

                x+=20
            y+=20
    dlimage=cv2.bitwise_and(image,image,mask=dlmask)
    cv2.imwrite(path+file[:-4]+' autoencoder'+file[-4:],dlimage)
    cv2.imwrite(path+file[:-4]+' mask'+file[-4:],mask)
    thresh=cv2.bitwise_and(image,image,mask=mask)
    cv2.imwrite(path+file[:-4]+' thresh'+file[-4:],thresh)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    for c in cnts:
        M = cv2.moments(c)
        try:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        except: continue
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)

    cv2.imwrite(path+file[:-4]+' output'+file[-4:],image)