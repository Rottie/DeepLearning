
# coding: utf-8

# In[1]:


#1.資料準備
#1A.讀取資料

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# In[2]:


#1B.讀取資料

print('train images     :', mnist.train.images.shape,
      'labels:'           , mnist.train.labels.shape)
print('validation images:', mnist.validation.images.shape,
      ' labels:'          , mnist.validation.labels.shape)
print('test images      :', mnist.test.images.shape,
      'labels:'           , mnist.test.labels.shape)


# In[3]:


#2.建立模型

#2A.建立 layer 函數

def layer(output_dim,input_dim,inputs, activation=None):
    W = tf.Variable(tf.random_normal([input_dim, output_dim]))
    b = tf.Variable(tf.random_normal([1, output_dim]))
    XWb = tf.matmul(inputs, W) + b
    if activation is None:
        outputs = XWb
    else:
        outputs = activation(XWb)
    return outputs


# In[4]:


#2B.建立輸入層 x 

x = tf.placeholder("float", [None, 784])


# In[5]:


#2C.建立隱藏層 h1

h1=layer(output_dim=1000,input_dim=784,
         inputs=x ,activation=tf.nn.relu)  


# In[6]:


#2D.建立輸出層

y_predict=layer(output_dim=10,input_dim=1000,
                    inputs=h1,activation=None)


# In[7]:


#3.定義訓練方式

#3A.建立訓練資料label真實值 placeholder

y_label = tf.placeholder("float", [None, 10])


# In[8]:


#3B.定義loss function

loss_function = tf.reduce_mean(
                  tf.nn.softmax_cross_entropy_with_logits
                         (logits=y_predict , 
                          labels=y_label))


# In[9]:


#3C.選擇optimizer

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)                     .minimize(loss_function)


# In[10]:


#4.定義評估模型的準確率

#4A.計算每一筆資料是否正確預測

correct_prediction = tf.equal(tf.argmax(y_label  , 1),
                              tf.argmax(y_predict, 1))


# In[11]:


#4B.將計算預測正確結果，加總平均

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# In[12]:


#5.開始訓練

#5A.定義訓練參數

trainEpochs = 15
batchSize = 100
totalBatchs = int(mnist.train.num_examples/batchSize)
epoch_list=[];loss_list=[];accuracy_list=[]
from time import time
startTime=time()
sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[13]:


#5B.進行訓練

for epoch in range(trainEpochs):
    for i in range(totalBatchs):
        batch_x, batch_y = mnist.train.next_batch(batchSize)
        sess.run(optimizer,feed_dict={x: batch_x,y_label: batch_y})
        
    loss,acc = sess.run([loss_function,accuracy],
                        feed_dict={x: mnist.validation.images, 
                                   y_label: mnist.validation.labels})

    epoch_list.append(epoch);loss_list.append(loss)
    accuracy_list.append(acc)    
    print("Train Epoch:", '%02d' % (epoch+1), "Loss=",                 "{:.9f}".format(loss)," Accuracy=",acc)
    
duration =time()-startTime
print("Train Finished takes:",duration)  


# In[14]:


#5C.畫出 loss 誤差執行結果

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
fig = plt.gcf()
fig.set_size_inches(4,2)
plt.plot(epoch_list, loss_list, label = 'loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss'], loc='upper left')


# In[15]:


#5D.畫出 accuracy 執行結果

plt.plot(epoch_list, accuracy_list,label="accuracy" )
fig = plt.gcf()
fig.set_size_inches(4,2)
plt.ylim(0.8,1)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()


# In[16]:


#6A.評估模型準確率

print("Accuracy:", sess.run(accuracy,
                           feed_dict={x: mnist.test.images, 
                                      y_label: mnist.test.labels}))


# In[17]:


#7.進行預測

#7A.執行預測

prediction_result=sess.run(tf.argmax(y_predict,1),
                           feed_dict={x: mnist.test.images })


# In[18]:


#7B.預測結果

prediction_result[:10]


# In[19]:


#7C.顯示前 10 筆預測結果

import matplotlib.pyplot as plt
import numpy as np
def plot_images_labels_prediction(images,labels,
                                  prediction,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num>25: num=25 
    for i in range(0, num):
        ax=plt.subplot(5,5, 1+i)
        
        ax.imshow(np.reshape(images[idx],(28, 28)), 
                  cmap='binary')
            
        title= "label=" +str(np.argmax(labels[idx]))
        if len(prediction)>0:
            title+=",predict="+str(prediction[idx]) 
            
        ax.set_title(title,fontsize=10) 
        ax.set_xticks([]);ax.set_yticks([])        
        idx+=1 
    plt.show()


# In[20]:


#7D.顯示前 10 筆預測結果

plot_images_labels_prediction(mnist.test.images,
                              mnist.test.labels,
                              prediction_result,0)

