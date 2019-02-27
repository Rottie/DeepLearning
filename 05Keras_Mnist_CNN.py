
# coding: utf-8

# # 1. Import Library

# In[1]:


from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
np.random.seed(10)


# # 資料預處理

# In[2]:


(x_Train, y_Train), (x_Test, y_Test) = mnist.load_data()


# In[3]:


x_Train4D=x_Train.reshape(x_Train.shape[0],28,28,1).astype('float32')
x_Test4D=x_Test.reshape(x_Test.shape[0],28,28,1).astype('float32')


# In[4]:


x_Train4D_normalize = x_Train4D / 255
x_Test4D_normalize = x_Test4D / 255


# In[5]:


y_TrainOneHot = np_utils.to_categorical(y_Train)
y_TestOneHot = np_utils.to_categorical(y_Test)


# # 建立模型

# In[6]:


from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D


# In[7]:


model = Sequential()


# In[8]:


model.add(Conv2D(filters=16,
                 kernel_size=(5,5),
                 padding='same',
                 input_shape=(28,28,1), 
                 activation='relu'))


# In[9]:


model.add(MaxPooling2D(pool_size=(2, 2)))


# In[10]:


model.add(Conv2D(filters=36,
                 kernel_size=(5,5),
                 padding='same',
                 activation='relu'))


# In[11]:


model.add(MaxPooling2D(pool_size=(2, 2)))


# In[12]:


model.add(Dropout(0.25))


# In[13]:


model.add(Flatten())


# In[14]:


model.add(Dense(128, activation='relu'))


# In[15]:


model.add(Dropout(0.5))


# In[16]:


model.add(Dense(10,activation='softmax'))


# In[17]:


print(model.summary())


# # 訓練模型

# In[18]:


model.compile(loss='categorical_crossentropy',
              optimizer='adam',metrics=['accuracy']) 


# In[19]:


train_history=model.fit(x=x_Train4D_normalize, 
                        y=y_TrainOneHot,validation_split=0.2, 
                        epochs=20, batch_size=300,verbose=2)


# In[20]:


import matplotlib.pyplot as plt
def show_train_history(train_acc,test_acc):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[test_acc])
    plt.title('Train History')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


# In[21]:


show_train_history('acc','val_acc')


# In[22]:


show_train_history('loss','val_loss')


# # 評估模型準確率

# In[23]:


scores = model.evaluate(x_Test4D_normalize , y_TestOneHot)
scores[1]


# # 預測結果

# In[24]:


prediction=model.predict_classes(x_Test4D_normalize)


# In[25]:


prediction[:10]


# # 查看預測結果

# In[26]:


import matplotlib.pyplot as plt
def plot_images_labels_prediction(images,labels,prediction,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num>25: num=25 
    for i in range(0, num):
        ax=plt.subplot(5,5, 1+i)
        ax.imshow(images[idx], cmap='binary')

        ax.set_title("label=" +str(labels[idx])+
                     ",predict="+str(prediction[idx])
                     ,fontsize=10) 
        
        ax.set_xticks([]);ax.set_yticks([])        
        idx+=1 
    plt.show()


# In[27]:


plot_images_labels_prediction(x_Test,y_Test,prediction,idx=0)


# # confusion matrix

# In[28]:


import pandas as pd
pd.crosstab(y_Test,prediction,
            rownames=['label'],colnames=['predict'])


# In[29]:


df = pd.DataFrame({'label':y_Test, 'predict':prediction})


# In[30]:


df[(df.label==5)&(df.predict==3)]

