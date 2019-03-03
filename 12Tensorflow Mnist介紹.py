
# coding: utf-8

# In[7]:


#1.在 Tensorflow 對 mnist 資料進行預先處理

#1A.匯入 Tensorflow 模組

import tensorflow as tf


# In[8]:


#1B.讀取 mnist 資料集模組

import tensorflow.examples.tutorials.mnist.input_data as input_data


# In[9]:


#1C.第一次下載 mnist 資料集模組

#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# In[10]:


#1D.讀取 mnist 資料

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# In[11]:


#1E.查看 mnist 資料

print('train',mnist.train.num_examples,
      ',validation',mnist.validation.num_examples,
      ',test',mnist.test.num_examples)


# In[12]:


#2.查看訓練資料

#2A.查看訓練資料1的 images 與 labels 部分

print('train images     :', mnist.train.images.shape,
      'labels:'           , mnist.train.labels.shape)


# In[13]:


#2B.查看第 0 筆 images 影像的長度

len(mnist.train.images[0])


# In[14]:


#2C.查看第 0 筆 images 影像的内容

mnist.train.images[0]


# In[15]:


#2D.定義 plos_image 函數顯示影像

import matplotlib.pyplot as plt
def plot_image(image):
    plt.imshow(image.reshape(28,28),cmap='binary')
    plt.show()


# In[16]:


#2E.執行 plot_image 函數

plot_image(mnist.train.images[0])


# In[17]:


#2F.查看訓練 Labels 資料

mnist.train.labels[0]


# In[18]:


#2G.使用 argmax 顯示數字

import numpy as np
np.argmax(mnist.train.labels[0])


# In[19]:


#3.查看多筆訓練資料 images 與 label

#3A.修改 plot_images_labels_prediction()函數

import matplotlib.pyplot as plt
def plot_images_labels_prediction(images,labels,
                                  prediction,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num>25: num=25 
    for i in range(0, num):
        ax=plt.subplot(5,5, 1+i)
        
        ax.imshow(np.reshape(images[idx],(28, 28)),            #轉換 images 欄位
                  cmap='binary')
            
        title= "label=" +str(np.argmax(labels[idx]))           #轉換 labels 欄位
        if len(prediction)>0:
            title+=",predict="+str(prediction[idx]) 
            
        ax.set_title(title,fontsize=10) 
        ax.set_xticks([]);ax.set_yticks([])        
        idx+=1 
    plt.show()


# In[20]:


#3B.查看訓練資料前 10筆資料

plot_images_labels_prediction(mnist.train.images,
                              mnist.train.labels,[],0)


# In[21]:


#3C.查看 validation 資料筆數

print('validation images:', mnist.validation.images.shape,
      'labels:'           , mnist.validation.labels.shape)


# In[22]:


#3D.查看 validation 資料

plot_images_labels_prediction(mnist.validation.images,
                              mnist.validation.labels,[],0)


# In[23]:


#3E.查看 test 資料筆數

print('test images:', mnist.test.images.shape,
      'labels:'           , mnist.test.labels.shape)


# In[24]:


#3F.查看 test 資料

plot_images_labels_prediction(mnist.test.images,
                              mnist.test.labels,[],0)


# In[25]:


#4.批次讀取資料 mnist

#4A.讀取批次資料

batch_images_xs, batch_labels_ys =      mnist.train.next_batch(batch_size=100)


# In[26]:


#4B.查看批次資料筆數

print(len(batch_images_xs),
      len(batch_labels_ys))


# In[27]:


#4C.查看批次資料

plot_images_labels_prediction(batch_images_xs,
                              batch_labels_ys,[],0)

