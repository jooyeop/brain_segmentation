# brain_segmentation


# Video_Object_Detection
U-net 모델을 활용한 brain segmentation 프로젝트

## 프로젝트 목적
사람의 뇌 MRI 데이터를 활용해 Segmentation으로 신경교종 종양 유형의 환자를 구별

## 프로젝트 배경
Segmentaion 기술에 대한 이해력 상승

## 연구 및 개발에 필요한 데이터 셋 소개
https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

1.kaggle에서 뇌 종양 환자의 데이터셋을 다운로드 받아 모델을 구현하였습니다.

2.국내 데이터를 통해 프로젝트를 하고 싶었지만, 의료데이터를 구하기 쉽지 않았습니다.


## 연구 및 개발에 필요한 기술 스택
### U-Net
1. Contraction Path(encoding) : 이미지의 context를 포착
2. Expansive Path(decoding) : feature amp을 upsampling 하여 포착한 이미지의 context를 feature map의 context와 결합한다.
  -> 이는 더욱 정확한 localization을 하는 역할
  
U-Net은 적은 데이터로 충분한 학습을 하기 위해 Data Augmentation을 사용
Data Augmentation이란 원래의 데이터를 부풀려서 더 좋은 성능을 만든다는 뜻
Data Augmentation이 중요한 이유
1. Preprocessing & augmentation 진행 시 성능 상승
2. 원본에 추가되는 개념으로 성능이 떨어지지 않음
3. 쉽고 패턴이 정해져 있음

      
```Python3
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
    
def decoder_block(inp,filters,concat_layer):
    x=Conv2DTranspose(filters,(2,2),strides=(2,2),padding='same')(inp)
    concat_layer=attention_block(inp,concat_layer)
    x=concatenate([x,concat_layer])
    x=conv_block(x,filters)
    return x
    
inputs=Input((256,256,3))
d1,p1=encoder_block(inputs,64)
d2,p2=encoder_block(p1,128)
d3,p3=encoder_block(p2,256)
d4,p4=encoder_block(p3,512)
b1=conv_block(p4,1024)
e2=decoder_block(b1,512,d4)
e3=decoder_block(e2,256,d3)
e4=decoder_block(e3,128,d2)
e5=decoder_block(e4,64,d1)
outputs = Conv2D(1, (1,1),activation="sigmoid")(e5)
model=Model(inputs=[inputs], outputs=[outputs],name='AttnetionUnet')

```


## 결과
종양의 Segmentation이 포착됨

![image](https://user-images.githubusercontent.com/97720878/188047307-bfaa863c-2745-46e2-acf5-6efd4ca613a3.png)
![image](https://user-images.githubusercontent.com/97720878/188047351-796cf6c1-b3a4-4284-86d7-c80b04c82dd0.png)
![image](https://user-images.githubusercontent.com/97720878/188047379-6113cacc-91bd-4ca8-8908-c38a0b190b02.png)
![image](https://user-images.githubusercontent.com/97720878/188047215-9b9f7652-0c07-40ff-9ac7-3ab79ceee21d.png)


## 한계점 및 해결 방안
국내 데이터를 활용하여 모델을 구축해볼 예정

다른 사람의 참고코드를 활용한것이 아닌 직접 모델을 만드는 프로젝트 진행 예정


참고 코드
https://www.kaggle.com/code/shashank069/brainmri-image-segmentation-attentionunet/notebook
