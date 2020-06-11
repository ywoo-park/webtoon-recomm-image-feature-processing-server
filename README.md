# Elasticsearch 기반 유사 이미지 검색 서비스 Tensorflow Imagenet Feature 처리 서버

## 목차
[1. 프로젝트에 대해서](#프로젝트에-대해서)

[2. 서비스 아키텍쳐](#서비스-아키텍쳐)

[3. 프로젝트 구조](#프로젝트-구조)

[4. 프로젝트 설명](#프로젝트-설명)

[5. 시작하기](#시작하기)

[6. API](#api)

[7. 핵심 코드](#핵심-코드)

## 프로젝트에 대해서

본 프로젝트는 **네이버 쇼핑박스 개발팀 인턴십 과제**입니다.  

- 프로젝트 기간 : 1/6 ~ 2/28
- 참여자 : 박소현, 박영우
- 동일 프로젝트 타 Repository 
  - 프로젝트 소개 : https://oss.navercorp.com/shopadboxintern/shopad-intern-image-searching
  - 유사 이미지 검색 서비스 Core 서버  : https://oss.navercorp.com/shopadboxintern/shopad-intern-image-searching-core-server
  - UI : https://oss.navercorp.com/shopadboxintern/shopad-intern-image-searching-client
  
## 서비스 아키텍쳐

<div>
  <img width="80%" alt="flask_arch" src="https://media.oss.navercorp.com/user/16727/files/f3991300-5735-11ea-814d-a5389d554809">
</div>

## 프로젝트 구조
model  
ㄴinception_v3  
app.py  
requirements.txt  

## 프로젝트 설명
이미지 URL을 활용하여 이미지의 TF 특징점을 뽑거나 Elasticsearch에 저장하는 역할을 한다.

### TF 특징점
Tensorflow Imagenet Inception_v3 모델을 기반으로 추출한 imagenet_feature

### Inception_v3
Inception-v3은 ImageNet 데이터베이스의 1백만 개가 넘는 이미지에 대해 훈련된 컨벌루션 신경망이다. 이 네트워크는 48개의 계층으로 구성돼 있으며, 이미지 입력 크기는 299x299 이다. Inception-v3는 이미지를 대표하는 다양한 특징들을 학습하였고, 그 결과 이미지를 1,000가지 사물 범주로 분류할 수 있는 기능을 갖추고 있다. 

참고 설명: https://datascienceschool.net/view-notebook/8d34d65bcced42ef84996b5d56321ba9/

## 시작하기 
* Python 3.x 이상

* 패키지 설치 
```
pip install numpy
pip install requests
pip install flask
pip install tensorflow==2.0.0
pip install tensorflow_hub
pip install keras==2.3.1
pip install elasticsearch
```

* Inception_v3 Model 다운로드 참고 : https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4

## API
### 이미지 TF 특징점 저장

| Method | Path             | Explanation                                 |
| ------ | ---------------- | ------------------------------------------- |
| POST   | /saveImagenetFeature | 이미지 URL을 사용하여 이미지의 TF 특징점을 뽑고, Elasticsearch의 "img_list"에 저장 |

* 사용성 : 스프링 프로젝트에서 직접 호출 됨. (스프링 프로젝트와 같은 Index를 사용하며, 스프링에서 먼저 이미지 정보를 "img_list"에 저장하면, 이후 Flask에서 저장된 이미지에 해당하는 TF 특징점을 추출하여 Update 해줌)

* 요청 헤더
```
Content-Type : application/json
```

* 요청 바디 예시 
```
{
 "item_id" : "4816029088",
 "img_url" : "https://shop-phinf.pstatic.net/20200210_262/1581301917288vHeVa_JPEG/105713_1.jpg?type=w640"
}
```

### 이미지 TF 특징점 + 일반 정보 저장 
| Method | Path             | Explanation                                 |
| ------ | ---------------- | ------------------------------------------- |
| POST    | /saveAllImageInfo | 이미지 URL을 사용하여 이미지의 TF 특징점을 뽑고, 이미지의 일반 정보(item_id, item_name, category_id, img_url)와 함께 ES의 "tf_img_list"에 저장 |

* 사용성 : 스프링 프로젝트와 독립적으로 ES의 다른 Index에 저장할 경우 사용 

* 요청 헤더
```
Content-Type : application/json
```

* 요청 바디 예시
```
{
 "item_id": "4816029088",
 "item_name": "10컬러 무지 남녀공용 오버핏 박시핏 라운드티 티셔츠 (S~3XL) 소녀나라",
 "category_id": "00000030",
 "img_url": "https://shop-phinf.pstatic.net/20200210_262/1581301917288vHeVa_JPEG/105713_1.jpg?type=w640"
}
```

### 이미지 TF 특징점 추출

| Method | Path             | Explanation                                 |
| ------ | ---------------- | ------------------------------------------- |
| GET    | /getImagenetFeature | 이미지 URL을 사용하여 이미지의 TF 특징점을 뽑고 반환 |

* 사용성 : ES에 없는 새로운 이미지 검색 시 사용. (현 스프링 프로젝트에서는 이미지 검색 시, 이미지에 해당하는 TF 특징점을 ES에서 받아오는 방식으로, ES에 저장된 이미지만 검색 가능)

* 요청 예시 
```
http://localhost:5000/getImagenetFeature?img_url=https://shop-phinf.pstatic.net/20200210_262/1581301917288vHeVa_JPEG/105713_1.jpg?type=w640
```

* 응답 바디 
```
{
  "imagenet_feature" : [
      0.0,
      0.0,
      0.0,
      99.2962417602539,
      31.41985321044922,
      38.70750427246094,
      0.0,
      146.83554077148438,
      0.9729382991790771,
       ...
      31.41985321044922,
      38.70750427246094
  ]	
}
```

## 핵심 코드 

```
# tfhub.dev handle 사용 모듈 정의
module_handle = "model/inception_v3/"

# 모듈 로드
module = hub.load(module_handle)

# Resize 299 x 299 x 3 shape tensor
img = tf.compat.v1.image.resize_image_with_pad(img, 299, 299)

# new axis 를 추가하여 Data type 을 uint8 에서 float32 로 변환
# float32 1 x 299 x 299 x 3 tensor (inception_v3 model 에서 요구하는 형태)
img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]

# 이미지 feature vector 계산
features = module(img)

# 배열에서 single-dimensional entries 제거
# type : numpy.ndarray, float32, 2048 columns
feature_vector = np.squeeze(features)
```
