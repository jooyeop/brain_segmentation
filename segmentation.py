import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from itertools import chain
import random
import cv2
from glob import glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Lambda, Conv2D, Conv2DTranspose,MaxPooling2D, concatenate,UpSampling2D,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator



patient = []
for i in os.listdir('../data/lgg-mri-segmentation/kaggle_3m/'): # patient list
    if i!= 'data.csv' and i!='README.md' : # data.csv and README.md are not patient list
        patient.append(i) # 모델 생성


def create_dataset(start,end,dataset_type): # dataset_type : train, val, test
    train_files = [] # train_files : train 데이터의 파일 리스트
    mask_files=[] # mask_files : train 데이터의 마스크 파일 리스트
    c=0
    for i,p in enumerate(patient[start:end]): # patient[start:end] : train 데이터의 파일 리스트
        vals=[] # vals : train 데이터의 파일 리스트
        mask_files.append(glob('../data/lgg-mri-segmentation/kaggle_3m/'+p+'/*_mask*')) # glob : 파일 리스트를 반환합니다. glob을 사용하면 파일의 경로를 입력하지 않아도 됩니다.
        for m in mask_files[i]: # mask_files[i] : train 데이터의 마스크 파일 리스트 
            vals.append(np.max(cv2.imread(m))) # np.max : 최대값을 반환합니다. 이유는 마스크 파일의 픽셀 값이 0이기 때문입니다.
        if max(vals)==0: # max : 리스트의 최대값을 반환합니다. 이유는 마스크 파일의 픽셀 값이 0이기 때문입니다.
            print(f'patient { p } has no tumor')
            c+=1 # C : 삭제된 환자의 수 카운트
    if c==0:
        print(f'Each patient in {dataset_type} dataset has brain tumor')
    mask_files=list(chain.from_iterable(mask_files)) # chain : 리스트의 요소를 연결합니다. 이유는 리스트의 요소들이 다른 리스트들이므로 연결하려면 리스트들을 연결해야합니다.
    for m in mask_files:
        train_files.append(m.replace('_mask','')) # replace : 문자열을 변경합니다. 이유는 마스크 파일의 픽셀 값이 0이기 때문입니다.
    df = pd.DataFrame(data={"filepath": train_files, 'mask' : mask_files}) # df : 데이터프레임을 생성합니다.
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
    p = np.random.randint(0, end) # p : 랜덤한 이미지의 인덱스
    img = cv2.imread(df_train[col].loc[p]) # cv2.imread : 이미지를 읽습니다.
    unique, counts = np.unique(img, return_counts=True) # np.unique : 중복된 요소를 제거합니다. 중복된 요소를 제거하면 요소의 개수를 반환합니다.
    print(f'showing pixel value counts for image {p}') 
    print(np.asarray((unique, counts)).T)

# pixel_value_counts('filepath',len(df_train)) 테스트 및 검증 셋에서 픽셀 값의 카운트를 보여줍니다.
pixel_value_counts('mask',len(df_train))


#마스크를 테스트 데이터에서 0과 1으로 표시합니다. 이것은 프로젝트를 마치기 전에 유용합니다.
for i in range(0,len(df_test)): 
    arr = np.where(cv2.imread(df_test['mask'].loc[i])==255,1,0) # np.where : 조건에 맞는 요소를 반환합니다.
    v = np.max(arr) # np.max : 리스트의 최대값을 반환합니다.
    if v==1:
        df_test.loc[i,'res'] = 1 # loc : 인덱스를 사용해서 열을 선택합니다.
    else:
        df_test.loc[i,'res'] = 0 # loc : 인덱스를 사용해서 열을 선택합니다.


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


#데이터 생성
def img_dataset(df_inp, path_img, path_mask, aug_args, batch) : # df_inp : 데이터프레임, path_img : 이미지 경로, path_mask : 마스크 경로, aug_args : 아담 파라미터, batch : 배치 크기
    img_gen = ImageDataGenerator(rescale=1./255., **aug_args) # ImageDataGenerator : 이미지 데이터 생성기입니다.
    df_img = img_gen.flow_from_dataframe(dataframe=df_inp,
                                            x_col=path_img,
                                            class_mode = None,
                                            batch_size=batch,
                                            color_mode='rgb',
                                            seed = 1,
                                            target_size = (256,256))

    df_mask = img_gen.flow_from_dataframe(dataframe=df_inp,
                                            x_col = path_mask,
                                            class_mode = None,
                                            batch_size=batch,
                                            color_mode='grayscale',
                                            seed = 1,
                                            target_size = (256,256))
    data_gen = zip(df_img, df_mask)
    return data_gen


# 손실 함수
def dice_loss(y_true, y_pred):
    y_true = K.flatten(y_true) 
    y_pred = K.flatten(y_pred)
    intersec = K.sum(y_true * y_pred) # K.sum : 리스트의 합을 반환합니다.
    return (-((2* intersec + 0.1) / (K.sum(y_true) + K.sum(y_pred) + 0.1))) # 0.1은 임의의 값입니다.

def iou(y_true, y_pred):
    intersec = K.sum(y_true * y_pred) # K.sum : 리스트의 합을 반환합니다.
    union = K.sum(y_true + y_pred) 
    iou = (intersec + 0.1) / (union - intersec + 0.1) 
    return iou


#모델
def conv_block(inp, filters):
    x = Conv2D(filters, (3,3), padding='same', activation='relu')(inp) # Conv2D : 2D 컨벌루션 레이어입니다.
    x = Conv2D(filters, (3,3), padding='same')(x) 
    x = BatchNormalization(axis = 3)(x) # BatchNormalization : 배치 노멀라이제이션 레이어입니다.
    x = Activation('relu')(x) # Activation : 활성화 함수를 적용합니다.
    return x

def encoder_block(inp, filters):
    x = conv_block(inp, filters) # conv_block : 컨벌루션 블록을 적용합니다.
    p = MaxPooling2D(pool_size=(2,2))(x) # MaxPooling2D : 2차원 맥스 풀링 레이어입니다.
    return x,p

def attention_block(l_layer, h_layer) :
    phi = Conv2D(h_layer.shape[-1],(1,1),padding='same')(l_layer) #phi : 입력 레이어의 출력을 입력 레이어의 출력과 같은 크기로 만듭니다.
    theta = Conv2D(h_layer.shape[-1],(1,1),strides=(2,2), padding='same')(h_layer) #theta : 입력 레이어의 출력을 입력 레이어의 출력과 같은 크기로 만듭니다.
    x = tf.keras.layers.add([phi, theta]) # tf.keras.layers.add : 두 레이어의 출력을 더합니다.
    x = Activation('relu')(x) # Activation : 활성화 함수를 적용합니다.
    x = Conv2D(1,(1,1),padding='same', Activation='sigmoid')(x)
    x = UpSampling2D(size = (2,2))(x)
    x = tf.keras.layers.muliply([h_layer, x])
    x = BatchNormalization(axis = 3)(x)
    return x

def decoder_block(inp, filters, concat_layer):
    x = Conv2DTranspose(filters, (2,2), strides=(2,2), padding='same')(inp) # Conv2DTranspose : 2차원 컨벌루션 방향 변환 레이어입니다.
    concat_layer = attention_block(inp, concat_layer) # attention_block : 어텐션 블록을 적용합니다.
    x = concatenate([x, concat_layer]) # concatenate : 두 레이어를 연결합니다.
    x = conv_block(x, filters) # conv_block : 컨벌루션 블록을 적용합니다.
    return x

def conv_block(inp,filters):
    x=Conv2D(filters,(3,3),padding='same',activation='relu')(inp)
    x=Conv2D(filters,(3,3),padding='same')(x)
    x=BatchNormalization(axis=3)(x)
    x=Activation('relu')(x)
    return x

def encoder_block(inp,filters):
    x=conv_block(inp,filters)
    p=MaxPooling2D(pool_size=(2,2))(x)
    return x,p

def attention_block(l_layer,h_layer): #Attention Block
    phi=Conv2D(h_layer.shape[-1],(1,1),padding='same')(l_layer) 
    theta=Conv2D(h_layer.shape[-1],(1,1),strides=(2,2),padding='same')(h_layer)
    x=tf.keras.layers.add([phi,theta])
    x=Activation('relu')(x)
    x=Conv2D(1,(1,1),padding='same',activation='sigmoid')(x)
    x=UpSampling2D(size=(2,2))(x)
    x=tf.keras.layers.multiply([h_layer,x])
    x=BatchNormalization(axis=3)(x)
    return x
    
def decoder_block(inp,filters,concat_layer): #Decoder Block
    x=Conv2DTranspose(filters,(2,2),strides=(2,2),padding='same')(inp) 
    concat_layer=attention_block(inp,concat_layer)
    x=concatenate([x,concat_layer])
    x=conv_block(x,filters)
    return x

inputs=Input((256,256,3)) #Input : 입력 레이어입니다.
d1,p1=encoder_block(inputs,64) #Encoder Block 1
d2,p2=encoder_block(p1,128) #Encoder Block 2
d3,p3=encoder_block(p2,256) #Encoder Block 3
d4,p4=encoder_block(p3,512) #Encoder Block 4
b1=conv_block(p4,1024) #Bottleneck 1
e2=decoder_block(b1,512,d4) #Decoder Block 2
e3=decoder_block(e2,256,d3) #Decoder Block 3
e4=decoder_block(e3,128,d2) #Decoder Block 4
e5=decoder_block(e4,64,d1) #Decoder Block 5
outputs = Conv2D(1, (1,1),activation="sigmoid")(e5) #Output Layer
model=Model(inputs=[inputs], outputs=[outputs],name='AttnetionUnet') #Model : 모델을 정의합니다.


model.summary()


# 훈련 세트에 대한 데이터 증대 작업 수행
augmentation_args = dict(rotation_range = 0.2,
                         width_shift_range = 0.05,
                        height_shift_range = 0.05,
                        shear_range = 0.05,
                        zoom_range = 0.05,
                        fill_mode = 'nearest') #앙상블 인자 정의

batch = 32


def train_model(model, save_name, loss_func):
    opt = Adam(learning_rate = 1e-4, epsilon = None, amsgrad = False, beta_1 = 0.9, beta_2 = 0.99) #Adam : 앙상블 옵티마이저입니다.
    model.compile(optimizer = opt, loss = loss_func, metrics = [iou]) 
    callbacks = [ModelCheckpoint(save_name, verbose = 1, save_best_only = True), 
                 ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, verbose = 1, patience = 5, min_lr = 1e-6), #ReduceLROnPlateau : 일정 에폭 수동 조절을 수행합니다.
                 EarlyStopping(monitor = 'val_loss', patience = 15, restore_best_weights=True)]  #EarlyStopping : 에러가 없을 때 학습을 중단합니다.
    train = img_dataset(df_train, 'filepath', 'mask', augmentation_args, batch) #img_dataset : 이미지 데이터셋을 정의합니다.
    val = img_dataset(df_val, 'filepath', 'mask', dict(), batch) #img_dataset : 이미지 데이터셋을 정의합니다.


    history = model.fit_generator(train,
                                    validation_data = val,
                                    steps_per_epoch = len(df_train)/batch,
                                    validation_steps = len(df_val)/batch,
                                    epochs = 25,
                                    callbacks = callbacks)


    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.tile('train loss')
    plt.legend(['train', 'val'], loc = 'upper left')
    plt.show()


train_model(model, 'unet_wts1.hdf5', dice_loss)


# 평가

def eval_model(model_wts,custom_objects):
    model = load_model(model_wts,custom_objects=custom_objects) #load_model : 모델을 불러옵니다.
    test=img_dataset(df_test[['filepath','mask']],'filepath','mask',dict(),32) #img_dataset : 이미지 데이터셋을 정의합니다.
    model.evaluate(test,steps=len(df_test)/32) #evaluate : 모델을 평가합니다.
    a=np.random.RandomState(seed=42) #np.random.RandomState : 랜덤 샘플링을 위한 샘플링 시드를 정의합니다.
    indexes=a.randint(1,len(df_test[df_test['res']==1]),10) #randint : 정수 난수를 생성합니다.
    for i in indexes:
        img = cv2.imread(df_test[df_test['res']==1].reset_index().loc[i,'filepath'])
        img = cv2.resize(img ,(256, 256))
        img = img / 255
        img = img[np.newaxis, :, :, :]
        pred=model.predict(img)

        plt.figure(figsize=(12,12))
        plt.subplot(1,3,1)
        plt.imshow(np.squeeze(img))
        plt.title('Original Image')
        plt.subplot(1,3,2)
        plt.imshow(np.squeeze(cv2.imread(df_test[df_test['res']==1].reset_index().loc[i,'mask'])))
        plt.title('Original Mask')
        plt.subplot(1,3,3)
        plt.imshow(np.squeeze(pred) > .5)
        plt.title('Prediction')
        plt.show()

eval_model('unet_wts1.hdf5',{'dice_loss':dice_loss,'iou':iou})