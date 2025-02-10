# DeepLearning-
Core concepts

Train과 test data로 나누는 이유

- 모델의 일반화 능력을 높이기 위해

Self-supervisor learning

- 비지도 학습의 방법 중 하나

Semi-supervisor learning

- Super 와 unsuper를 섞는 것, 모델을 섞을 수도, data를 섞을수도

마스킹과 노이징 하는 이유 (41)

- ㄹ

Feature selection, Feature engineering

- Selection: 불필요한 특성 제거, 성능을 높이기 위해
- Engineering: feature를 새롭게 만들거나 엔지니어링 하는 것, 특징 추출해 가공하는 것

Overfitting

- 왜 발생: 데이터에 비해 모델이 복잡하면 발생
- 해결: norm1, norm2
- Norm1: 필요 없는 것을 없애는 것, Norm2: 0으로 가깝게 만드는 것

Underfitting

- 왜 발생: 데이터에 비해 모델이 간단하면 발생
- 해결: 데이터를 추가, argumentation

Embedding

- 왜 하는지: 모델은 문자를 이해를 못해서 숫자로 바꿔야한다
- 방법: one hot encoding, vector encoding
- Oen hot encoding 하는 이유: 순서의 관계가 모델에 영향을 미친다

앙상블

- 모델을 섞어 쓰는 것

Random forest

- 샘플을 boot strapping, 중복을 허락해서 여러 개로 만든다 -> 좋은 것을 선택
- 중복을 허락하지 않는 것: boot strapping aggregation

Hyper parameter tuning

- Hyper parameter 값을 변경하는데 변경하는 이유는
- 목적: 모델의 성능을 높이기 위해, 어떻게?
- 최적의 성능을 찾아주기 위해

Confusion matrix

- 어느 class에서 분류가 잘 안됐는지 알기 위해서 사용
- F(1)값을 사용하는 이유: 데이터 불균형

Precision, recall (151)

- 사용하는 이유:
- Precision 줄이려면 FP (force positive)를 줄임
- Recall 줄이려면 FN (force negative)를 줄임

경상화법

- 목적: 실제 값과 예측 값의 오차를 최소화 하기위해 사용
- 3가지: 확률적, 배치화, 미니배치
- 확률적 경상화법
- 장점: 시간이 빠르다
- 단점: 정확도가 떨어진다
- 배치화 경상화법
- 장점: 전수 조사하면 정확하다
- 단점: 시간이 느림
- 미니배치 (grouping)
- 확률적, 배치화의 하이브리드 방법

Support Vector Machine

- 결정 경계 근처 있는 애들

차원의 저주

- Feature가 너무 많아서 계산하기 어려워진다

PCA

- 목적: 정보를 압축시켜 차원을 압축시켜 주 성분을 찾는다

Manifold가설

- 고차원의 데이터도 저차원으로 표현할 수 있다. 저차원으로 표현하면 학습이 쉬워진다.

Activation function

- 사용하는 이유: 각 층마다 비선형성을 줌으로써 복잡한 특성을 추출할 수 있다.

<Gradient 소실, 폭주 현상, 해결방법>

Pre-trained model

- 사용 목적: 남이 잘 만들어 놓은 모델을 가져다 사용한다, 학습량 줄일 수 있다,
- Fine tuning

GAN, Auto encoder, variation 시험 문제

Variation autoencoder

- 정보 생성, latent space로부터 정보 생성

GAN

- 확률분포를 예측하는 것
- 장점: mode collapse가 발생
- Mode collapse: 다양한 출력을 못하고 특정 패턴이나 모드에만 집중해 같은 결과만 반복해서 출력하는 것
- 단점: 다양한 데이터를 생성하지 못한다

Decoder

- 데이터 복원, latent vector의 의미 변환, new data 생성

Encoder

- 데이터 압축, 특징 추출

Auto Encoder

- 사용 목적: 차원 축소
- 어떻게 축소? Input 데이터를 받아 encoder해서 latent vector로 만든다.

Latent vector vs Latent space

- Latent vector: 점 하나로 매칭
- Latent space: 공간에서 매칭, 특징들을 압축압축해서 모아놓은 집합

CNN

- 기본적인 차이점: 인접 셀, local적인 feature들을 추출가능

Pooling 목적

- 정보 요약, 이동성 불변

Resnet

- 주요 차이점: 사전 정보가 있는 것과 없는 것의 차이는 크다.

Inception

- Skip connection 사용, 서로 다른 필터를 동시에 사용해 학습을 한다.

One by one

- 목적: 정보 축약

Depth wise convolution

- 각 채널마다 별도의 필터를 사용해 채널 간의 상호작용은 고려되지 않고 각 채널은 독립적으로 처리

Parameter 개수가 많아 질수록 학습이 쉬워지는데 무거워진다. -> 어떻게 경량화 하면서 성능을 높일 것인가?

CNN 과 RNN의 차이

- CNN은 현재의 상태만 반영함, 문제점: 긴 연산에 대해서는 못한다. (ex. 주식)
- RNN은 현재의 정보만 받을 뿐만 아니라 이전 상태의 정보도 반영한다

CNN

- 로컬 정보 수집

RNN

- 목적:
- 현재 정보를 받을 뿐만 아니라 이전 상태의 정보도 반영

LSTM

- 목적:
- 롱텀의 주요 정보들, 중요 정보들만 남긴다
- 단점: gradient benet

ResNet 특징

- Skip connection -> 장점: GVP 해결 가능

SeNet

- 중요한 채널은 집중
- 채널의 중요도를 계산

Inception구조

- 서로 다른 필터 적용해 다양한 특징을 한꺼번에 추출해서 connect

Depth wise convolution

- 각 채널마다 별도의 필터를 사용하고 합치는 과정

Transformer

- Attention = 중요도
- Embedding: 숫자 변환
- Positional: 위치 지정
- Self:
- Cross: decoder와 encoder의 관계
- Transformer은 sequence to sequence model이다.
- Sequence to sequence 모델은 decoder와 encoder로 이루어져 있다.

Scale dot attention

- 전체 스코어에 벡터의 크기를 나누는 것, 전체 크기를 맞춰주는 것

Vision transformer

- 패치 구분해서 녹음(55:20)

인공지능 vs 딥러닝

- AI는 많은 알고리즘을 표현
- 머신러닝: 입력과 출력으로부터 규칙을 얻어낸다. 데이터의 특성을 발견해서 출력을 예측. 입력 데이터의 분포를 보고 출력 예측.
- 딥러닝은 신경망 내에서 추출과 분류, 입력과 출력으로부터 f(x)를 구하는 것
