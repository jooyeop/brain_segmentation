import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from itertools import chain
import random
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Lambda, Conv2D, Conv2DTranspose,MaxPooling2D, concatenate,UpSampling2D,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator



patient = []
for i in os.listdir('../data/lgg-mri-segmentation/kaggle_3m/'):
    if i!= 'data.csv' and i!='README.md' :
        patient.append(i)


def create_dataset(start,end,dataset_type):
    train_files = []
    mask_files=[]
    c=0
    for i,p in enumerate(patient[start:end]):
        vals=[]
        mask_files.append(glob('../data/lgg-mri-segmentation/kaggle_3m/'+p+'/*_mask*'))
        for m in mask_files[i]:
            vals.append(np.max(cv2.imread(m)))
        if max(vals)==0:
            print(f'patient { p } has no tumor')
            c+=1
    if c==0:
        print(f'Each patient in {dataset_type} dataset has brain tumor')
    mask_files=list(chain.from_iterable(mask_files))
    for m in mask_files:
        train_files.append(m.replace('_mask',''))
    df = pd.DataFrame(data={"filepath": train_files, 'mask' : mask_files})
    return df


# 트레인 데이터, 테스트 데이터, 검증 데이터의 길이
a = int(0.9*len(patient))
b = int(0.8*a)



# 주어진 데이터에서 있는 환자중에서 있는 환자가 없는 환자를 찾아봅니다.
df_train=create_dataset(0,b,'training')
df_val=create_dataset(b,a,'validation')
df_test=create_dataset(a,len(patient),'testing')


#기능 : 이미지의 픽셀 범위를 보는 함수. 메모 : 이 함수는 매번 호출될 때 랜덤한 이미지를 선택합니다.
def pixel_value_counts(col, end):
    p = np.random.randint(0, end)
    img = cv2.imread(df_train[col].loc[p])
    unique, counts = np.unique(img, return_counts=True)
    print(f'showing pixel value counts for image {p}')
    print(np.asarray((unique, counts)).T)

# pixel_value_counts('filepath',len(df_train)) 테스트 및 검증 셋에서 픽셀 값의 카운트를 보여줍니다.
pixel_value_counts('mask',len(df_train))


#마스크를 테스트 데이터에서 0과 1으로 표시합니다. 이것은 프로젝트를 마치기 전에 유용합니다.
for i in range(0,len(df_test)):
    arr = np.where(cv2.imread(df_test['mask'].loc[i])==255,1,0)
    v = np.max(arr)
    if v==1:
        df_test.loc[i,'res'] = 1
    else:
        df_test.loc[i,'res'] = 0


# 랜덤 환자에 대한 MRI 스캔 이미지를 보여줍니다.
f,ax=plt.subplots(3,3,figsize=(14,8))
ax=ax.flatten()
for j in range(0,9):
    i=1453+j
    img=cv2.imread(df_train['filepath'].loc[i])
    msk=cv2.imread(df_train['mask'].loc[i])
    ax[j].imshow(msk)
    ax[j].imshow(img,alpha=0.7)
plt.show()