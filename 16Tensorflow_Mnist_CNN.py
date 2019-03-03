
# coding: utf-8

# In[1]:


#1.資料準備
#1A .讀取資料

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# In[2]:


#2.建立共用函數

#2A.定義 weight 函數,用於建構權重 weight 張量
def weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1),
                       name ='W')

#2B.定義 bias 函數,建立偏差 bias 張量
def bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape)
                       , name = 'b')

#2c.定義 conv2d函數,進行卷積運算
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], 
                        padding='SAME')

#2D.建立 max_pool_2x2 函數,用於建立池化層
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], 
                          strides=[1,2,2,1], 
                          padding='SAME')


# In[3]:


#3.建立模型

#3A.輸入層 Input Layer

with tf.name_scope('Input_Layer'):
    x = tf.placeholder("float",shape=[None, 784]
                       ,name="x")    
    x_image = tf.reshape(x, [-1, 28, 28, 1])

#3B.卷積層 1 Convolutional Layer 1
with tf.name_scope('C1_Conv'):
    W1 = weight([5,5,1,16])
    b1 = bias([16])
    Conv1=conv2d(x_image, W1)+ b1
    C1_Conv = tf.nn.relu(Conv1 )

#3C.建立池化層 1 pooling layer 1
with tf.name_scope('C1_Pool'):
    C1_Pool = max_pool_2x2(C1_Conv)

#3D.卷積層 2 Convolutional Layer 2
with tf.name_scope('C2_Conv'):
    W2 = weight([5,5,16,36])
    b2 = bias([36])
    Conv2=conv2d(C1_Pool, W2)+ b2
    C2_Conv = tf.nn.relu(Conv2)

#3E.建立池化層 2 pooling layer 2
with tf.name_scope('C2_Pool'):
    C2_Pool = max_pool_2x2(C2_Conv) 
    
#3F.建立平坦層 Flatten layer
with tf.name_scope('D_Flat'):
    D_Flat = tf.reshape(C2_Pool, [-1, 1764])

#3G.建立隱藏層 Hidden Layer
with tf.name_scope('D_Hidden_Layer'):
    W3= weight([1764, 128])
    b3= bias([128])
    D_Hidden = tf.nn.relu(
                  tf.matmul(D_Flat, W3)+b3)
    D_Hidden_Dropout= tf.nn.dropout(D_Hidden, 
                                keep_prob=0.8)

#3H.建立輸出層
with tf.name_scope('Output_Layer'):
    W4 = weight([128,10])
    b4 = bias([10])
    y_predict= tf.nn.softmax(
                 tf.matmul(D_Hidden_Dropout,
                           W4)+b4)


# In[4]:


#4.定義訓練方式


with tf.name_scope("optimizer"):

#4A.建立訓練資料label真實值 placeholder
    y_label = tf.placeholder("float", shape=[None, 10], 
                              name="y_label")

#4B.定義loss function
    loss_function = tf.reduce_mean(
                      tf.nn.softmax_cross_entropy_with_logits
                         (logits=y_predict , 
                          labels=y_label))
#4C.選擇optimizer    
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)                     .minimize(loss_function)


# In[5]:


#5.定義評估模型的準確率方式

with tf.name_scope("evaluate_model"):
    correct_prediction = tf.equal(tf.argmax(y_predict, 1),
                                  tf.argmax(y_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# In[6]:


#6.進行訓練
#6A.定義訓練參數

trainEpochs = 30
batchSize = 100
totalBatchs = int(mnist.train.num_examples/batchSize)
epoch_list=[];accuracy_list=[];loss_list=[];
from time import time
startTime=time()
sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[7]:


#6B.進行訓練     (CPU 3640)

for epoch in range(trainEpochs):

    
    for i in range(totalBatchs):
        batch_x, batch_y = mnist.train.next_batch(batchSize)
        sess.run(optimizer,feed_dict={x: batch_x,
                                      y_label: batch_y})
        
    
    loss,acc = sess.run([loss_function,accuracy],
                        feed_dict={x: mnist.validation.images, 
                                   y_label: mnist.validation.labels})

    epoch_list.append(epoch)
    loss_list.append(loss);accuracy_list.append(acc)    
    
    print("Train Epoch:", '%02d' % (epoch+1),           "Loss=","{:.9f}".format(loss)," Accuracy=",acc)
    
duration =time()-startTime
print("Train Finished takes:",duration)  


# In[8]:


#6C.畫出 loss 結果

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
fig = plt.gcf()
fig.set_size_inches(4,2)
plt.plot(epoch_list, loss_list, label = 'loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss'], loc='upper left')


# In[9]:


#6D.畫出 accuracy結果

plt.plot(epoch_list, accuracy_list,label="accuracy" )
fig = plt.gcf()
fig.set_size_inches(4,2)
plt.ylim(0.8,1)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()


# In[10]:


#7.評估模型準確率

print("Accuracy:", 
      sess.run(accuracy,feed_dict={x: mnist.test.images,
                                   y_label: mnist.test.labels}))


# In[11]:


#8.進行預測
#8A.執行預測

prediction_result=sess.run(tf.argmax(y_predict,1),
                           feed_dict={x: mnist.test.images ,
                                      y_label: mnist.test.labels})


# In[12]:


#8B.預測結果

prediction_result[:10]


# In[13]:


#8C.顯示前 10 筆資料預測結果

import numpy as np
def show_images_labels_predict(images,labels,prediction_result):
    fig = plt.gcf()
    fig.set_size_inches(8, 10)
    for i in range(0, 10):
        ax=plt.subplot(5,5, 1+i)
        ax.imshow(np.reshape(images[i],(28, 28)), 
                  cmap='binary')
        ax.set_title("label=" +str(np.argmax(labels[i]))+
                     ",predict="+str(prediction_result[i])
                     ,fontsize=9) 
    plt.show()


# In[14]:


#8D.顯示前 10 筆資料預測結果

show_images_labels_predict(mnist.test.images,mnist.test.labels,prediction_result)


# In[22]:


#9.Tensorboard

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('log/CNN',sess.graph)

#查看檔案是否發生
#dir C:\NLP\log\CNN

#activate tensorflow-gpu

#tensorboard --logdir=C:\NLP\log\CNN

