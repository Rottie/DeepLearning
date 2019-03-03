
# coding: utf-8

# In[1]:


#1.Tensorflow 張量 運算模擬神經網路

import tensorflow as tf
import numpy as np


# In[2]:



#1A.矩陣運算加入 relu  激活函數
#  y= relu ( (X․W ) + b )

X = tf.Variable([[0.4,0.2,0.4]])

W = tf.Variable([[-0.5,-0.2 ],
                 [-0.3, 0.4 ],
                 [-0.5, 0.2 ]])
                         
b = tf.Variable([[0.1,0.2]])
    
XWb =tf.matmul(X,W)+b

y=tf.nn.relu(tf.matmul(X,W)+b)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print('XWb:')    
    print(sess.run(XWb ))    
    print('y:')    
    print(sess.run(y ))


# In[3]:


#1B.矩陣運算加入 sigmoid 激活函數
# y= sigmoid ( (X․W ) + b )

X = tf.Variable([[0.4,0.2,0.4]])

W = tf.Variable([[-0.5,-0.2 ],
                 [-0.3, 0.4 ],
                 [-0.5, 0.2 ]])

b = tf.Variable([[0.1,0.2]])

XWb=tf.matmul(X,W)+b

y=tf.nn.sigmoid(tf.matmul(X,W)+b)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print('XWb:')    
    print(sess.run(XWb))    
    print('y:')    
    print(sess.run(y ))


# In[4]:


#1C.以亂數產生Weight(W)與bais(b)

#用來建立多層感知器(MLP),工具是tensorflow提供 tf.random normal
#個別執行 3個變數

W = tf.Variable(tf.random_normal([3, 2]))
b = tf.Variable(tf.random_normal([1, 2]))
X = tf.Variable([[0.4,0.2,0.4]])
y=tf.nn.relu(tf.matmul(X,W)+b)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print('b:')
    print(sess.run(b ))    
    print('W:')
    print(sess.run(W ))
    print('y:')
    print(sess.run(y ))    


# In[5]:


#執行一次 sess.run 取得 3 個變數

W = tf.Variable(tf.random_normal([3, 2]))
b = tf.Variable(tf.random_normal([1, 2]))
X = tf.Variable([[0.4,0.2,0.4]])
y=tf.nn.relu(tf.matmul(X,W)+b)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    (_b,_W,_y)=sess.run((b,W,y))
    print('b:')
    print(_b)
    print('W:')
    print(_W)
    print('y:')
    print(_y)   


# In[18]:


#1D.常態分佈亂數簡單程式碼説明
#以 tf.random_normal 產生-->常態分佈亂數的 list

ts_norm = tf.random_normal([1000])
with tf.Session() as session:
    norm_data=ts_norm.eval()
print(len(norm_data))
print(norm_data[:5])


# In[10]:


#以plt.hist 顯示常態分佈圖形

import matplotlib.pyplot as plt
plt.hist(norm_data)
plt.show()    


# In[6]:


#2A.以 placeholder傳入 x 值
#運算 1x3 二維陣列

W = tf.Variable(tf.random_normal([3, 2]))
b = tf.Variable(tf.random_normal([1, 2]))
X = tf.placeholder("float", [None,3])     #定義 Placeholder
y=tf.nn.relu(tf.matmul(X,W)+b)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    X_array = np.array([[0.4,0.2,0.4]])  #建立 X_array
    (_b,_W,_X,_y)=sess.run((b,W,X,y),feed_dict={X:X_array}) #執行 sess.run
    print('b:')
    print(_b)    
    print('W:')
    print(_W)
    print('X:')
    print(_X)
    print('y:')
    print(_y)


# In[11]:


#運算 3x3 二維陣列

W = tf.Variable(tf.random_normal([3, 2]))
b = tf.Variable(tf.random_normal([1, 2]))
X = tf.placeholder("float", [None,3])
y=tf.nn.sigmoid(tf.matmul(X,W)+b)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    X_array = np.array([[0.4,0.2 ,0.4],
                        [0.3,0.4 ,0.5],
                        [0.3,-0.4,0.5]])    
    (_b,_W,_X,_y)=sess.run((b,W,X,y),feed_dict={X:X_array})
    print('b:')
    print(_b)    
    print('W:')
    print(_W)
    print('X:')
    print(_X)
    print('y:')
    print(_y)


# In[12]:


#3.建立 Layer 函數以矩陣運算模擬神經網路

#3A.建立 layer函數(建立 2層神經網路)

def layer(output_dim,input_dim,inputs, activation=None):
    W = tf.Variable(tf.random_normal([input_dim, output_dim]))
    b = tf.Variable(tf.random_normal([1, output_dim]))
    XWb = tf.matmul(inputs, W) + b
    if activation is None:
        outputs = XWb
    else:
        outputs = activation(XWb)
    return outputs


# In[14]:


#3B.使用 layer 函數建立 3層類神經網路 (顯示 X,H,Y)

X = tf.placeholder("float", [None,4])
h=layer(output_dim=3,input_dim=4,inputs=X,
        activation=tf.nn.relu)
y=layer(output_dim=2,input_dim=3,inputs=h)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    X_array = np.array([[0.4,0.2 ,0.4,0.5]])    
    (layer_X,layer_h,layer_y)=             sess.run((X,h,y),feed_dict={X:X_array})
    print('input Layer X:')
    print(layer_X)
    print('hidden Layer h:')
    print(layer_h)
    print('output Layer y:')
    print(layer_y)


# In[15]:


#4.建立 layer_debug 函數顯示 bias與 weight

#4A.建立 layer_debug 函數 (回傳 output,w , b)


def layer_debug(output_dim,input_dim,inputs, activation=None):
    W = tf.Variable(tf.random_normal([input_dim, output_dim]))
    b = tf.Variable(tf.random_normal([1, output_dim]))
    XWb = tf.matmul(inputs, W) + b
    if activation is None:
        outputs = XWb
    else:
        outputs = activation(XWb)
    return outputs,W,b


# In[16]:


#4B.用 layer_debug 函數, 建立 3 層類神經網路(顯示 w , b

X = tf.placeholder("float", [None,4])
h,W1,b1=layer_debug(output_dim=3,input_dim=4,inputs=X,
                    activation=tf.nn.relu)
y,W2,b2=layer_debug(output_dim=2,input_dim=3,inputs=h)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    X_array = np.array([[0.4,0.2 ,0.4,0.5]])    
    (layer_X,layer_h,layer_y,W1,b1,W2,b2)=              sess.run((X,h,y,W1,b1,W2,b2),feed_dict={X:X_array})
    print('input Layer X:')
    print(layer_X)
    print('W1:')
    print(  W1)    
    print('b1:')
    print(  b1)    
    print('hidden Layer h:')
    print(layer_h)    
    print('W2:')
    print(  W2)    
    print('b2:')
    print(  b2)    
    print('output Layer y:')
    print(layer_y)

