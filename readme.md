# ImageCaption With Attention
## Feature
Deeplearning 분야에서 멀티 모달리티 integration은 갈수록 중요해질 것입니다. Image Captioning은 Computer Vision(CV). Natural Language Processing(NLP)의 사이에서 다리역할을 해주는 중요한 기술입니다.  

## Dataset
- COCO data set 2017 version
- 15만장 이상의 이미지, 5개 caption  
[COCO data](https://cocodataset.org/#home)  
***COCO dataset 용량에 주의하세요!***

## Files
- project4_v1_run.ipynb : model.py, utils.py가 필요합니다.
- practice_code.ipynb : 주석이 많이 달려있어서 모델 공부하기 좋습니다.(지저분,,model.py, utils.py 필요 없습니다.)   


## Model
### Image Captioning with Semantic Attention
![image](https://user-images.githubusercontent.com/74405346/116520266-78b92280-a90d-11eb-9d23-a5e9512d69fe.png)

Image Captioning의 접근 방식은 크게 ‘Top-Down Approach’와 ‘Bottom-Up Approach’로 구분됩니다. Top-Down Approach는 이미지를 통째로 CNN에 통과시켜서 얻은 ‘Feature’를 텍스트로 변환하는 반면 
Bottom-Up Approach에서는 이미지의 다양한 부분들로부터 단어들을 도출해내고, 이를 결합하여 문장을 얻어내는 방식입니다. 
Top-Down Approach가 현재 가장 많이 쓰이고 있는 접근 이며, RNN를 이용하여 각 Parameter들을 Train Data로부터 학습시킬 수 있기 때문에 성능기 뛰어납니다.  

하지만, 이러한 Top-Down Approach의 단점은 이미지의 디테일한 부분들에 집중하는 것이 'Bottom-Up Approach'에 비해 상대적으로 어렵다는 점입니다. 그래서 최근에는 NLP에서 쓰이는 Attention을 이용하여 이미지의 특히 중요한 feature에 집중하도록 하는 방식이 쓰입니다.

먼저 image feature는 InceptionV3과 같은 pre-trained model을 통해 image feature를 추출합니다. image feature를 LSTM에 넣어주고, 각 시점에서 LSTM의 Hidden state와 image feature를 활용해 Attention Context vector를 적용해 학습합니다.

## Output
![image](https://user-images.githubusercontent.com/74405346/116521394-ddc14800-a90e-11eb-868d-e00644e64980.png)


## Reference
https://openaccess.thecvf.com/content_cvpr_2016/papers/You_Image_Captioning_With_CVPR_2016_paper.pdf  
https://www.tensorflow.org/tutorials/text/image_captioning  
https://github.com/amanmohanty/idl-nncompress  
