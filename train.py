'''

training the first layer of the autoencoder

'''

import tensorflow as tf
import AutoEncoder as model
import os
import cv2
import csv

print('Loading images')

path1='./Training Data - BG/'
path2='./Training Data - Cells/'

images=[]
labels=[]

for i in [1,2,3,4,5,21,22,23,24,25,26,27,28,29,30]:
    path=path1+'H&E ('+str(i)+')/'
    files=os.listdir(path)
    for file in files:
        if file=='Ground_Truth_BG.jpg': continue
        image=cv2.imread(path+file)
        image=cv2.resize(image,(61,61),interpolation = cv2.INTER_CUBIC)
        images.append(image)
        labels.append([1,0])
    path=path2+'H&E ('+str(i)+')/'
    files=os.listdir(path)
    for file in files:
        if file=='Ground_Truth.jpg': continue
        image=cv2.imread(path+file)
        image=cv2.resize(image,(61,61),interpolation = cv2.INTER_CUBIC)
        images.append(image)
        labels.append([0,1])

print('Images loaded')

epoch=1000
trainsize=6000
batchsize=200
lr=0.1

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    model.restore(sess)
    for epoch in range(1,epoch+1):
        epochloss=0
        if epoch==500: lr=0.01
        if epoch==750: lr=0.001
        for i in range(1,int((trainsize/batchsize)+1)):
            batchimages,batchlabels=images[batchsize*(i-1):batchsize*i],labels[batchsize*(i-1):batchsize*i]
            loss,_=sess.run([model.loss,model.optimize],feed_dict={model.x:batchimages, model.y:batchlabels, model.learning_rate:lr})
            epochloss+=loss
        print('Epoch',epoch,'comleted, Epoch Loss :',epochloss)
        if epochloss==0: break
        model.save(sess)
        with open('layer3loss.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([epochloss])
