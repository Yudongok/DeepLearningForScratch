# CNN

# CNN 네트워크 구조

![image.png](image.png)

- 기존 Affine-ReLU연결 방식이 Conv-ReLU-Pooling으로 변경
- 출력에 가까운 층에선 Affine-ReLU구성
- 마지막 출력계층은 Affine-Softmax

**CNN**은 보통 '컨볼루션-풀링' 계층들과 '완전 연결' 계층들의 조합으로 이루어집니다.

1. **컨볼루션 계층 (Convolutional Layer)**
    - 입력으로 3차원 데이터(채널, 높이, 너비)를 받는다.
    - 필터 연산을 통해 **특징 맵(Feature Map)**이라는 3차원 데이터를 출력한다.
    - 코드의 `conv_output_size`는 이 특징 맵의 높이와 너비 크기를 계산하는 부분이다.
    `(input_size - filter_size + 2*filter_pad) / filter_stride + 1` 공식에 따라 계산되죠.
2. **풀링 계층 (Pooling Layer)**
    - 컨볼루션 계층에서 나온 특징 맵의 크기를 줄여(sub-sampling) 연산량을 감소시키고 주요 특징을 강조한다.
    - 코드에서는 `conv_output_size/2`를 통해 풀링(아마도 2x2 풀링) 후의 크기를 암시하고 있다.
3. **완전 연결 계층 (Fully-Connected Layer, Affine Layer)**
    - 분류와 같은 최종 작업을 수행하기 위해 사용된다.
    - 하지만 이 계층은 1차원 배열 형태의 데이터만 입력으로 받을 수 있다.

# CNN용어

- 특징 맵: 합성곱 계층의 입출력 데이터
- 합성곱 연산: 이미지 처리에서 말하는 필터 연산에 해당한다.
- 패딩(Padding): 입력데이터 주변을 특정값(0)으로 채움 → 주로 출력 크기를 조정할 목적으로 사용
- 스트라이드(Stride): 필터를 적용하는 위치의 간격 → 스트라이드를 키우면 출력이 작아짐
- 풀링: 세로, 가로 방향의 공간을 줄이는 연산. 최대 풀링, 평균 풀링이 있음. 이미지 인식 분야에서는 보통 최대 풀링을 사용.

<aside>
💡

3차원의 합성곱 연산에선 입력 데이터의 채널 수와 필터의 채널 수가 같아야 한다.

</aside>

---

# CNN 개념

## 블록으로 생각하기

채널수 = C, 높이 = H, 너비 = W, 필터 높이 = FH, 필터 너비 = FW

입력 데이터의 형상(C, H, W)

필터(C, FH, FW)

→ 필터가 1개일 경우 출력 데이터도 1개임. 출력 데이터 N개가 필요하면 필터를 N개로 늘리면 됨

필터 (FN, C, FH, FW)

## 배치 처리

N개의 데이터를 배치처리 할 경우 입력 데이터의 차원은 (N, C, H, W)로 바뀜.

출력 데이터의 차원은 (N, FN, OH, OW)가 됨.

여기서 4차원 데이터가 하나 흐를 때마다 데이터 N개에 대한 합성곱 연산이 이루어진다. 즉, N회분의 처리를 한번에 수행한다.

## im2col로 데이터 전개하기

im2col이란 입력 데이털르 필터링(가중치 계산)하기 좋게 전개하는 함수이다.

1. 입력 데이터에서 필터를 적용하는 영역(3차원 블록)을 한 줄로 늘어놓는다.
2. 입력 데이터를 전개한 다음에는 합성곱 계층의 필터(가중치)를 1열로 전개하고, 두 행렬의 곱을 계산하면 된다.
3. im2col방식으로 출력한 결과는 2차원 행렬이다. CNN은 데이터를 4차원 배열로 저장하므로 2차원인 출력 데이터를 4차원으로 변형(reshape)한다.

### im2col 실제 사용해보기

```python
im2col(input_data, filter_h, filter_w, stride=1, pad=0)
```

- input_data - (데이터수, 채널수, 높이, 너비)의 4차원 배열
- filter_h - 필터의 높이
- filter_w - 필터의 너비
- stride - 스트라이드
- pad - 패딩

```python
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W= W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        # 출력 크기 계산
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2*self.pad - FW) / self.stride)

				# 입력 데이터를 im2col로 전개
        col = im2col(x, FH, FW, self.stride, self.pad)
        # reshape에서 두 번쨰 인수를 -1로 지정하면 다차원 배열의 원소 수가 변환 후에도 똑같이
        # 유지되도록 적절히 묶어줌
        col_W = self.W.reshape(FN, -1).T
        out = np.dot(col, col_W) + self.b
        # transpose함수는 다차원 배열의 축 순서를 바꿔주는 함수이다.
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        return out
```

- (10, 3, 5, 5)형상을 한 다차원 배열 w의 원소 수는 총 750개이다. 이 배열에 reshape(10, -1)을 호출하면 750개의 원소를 10묶음으로, 즉 형상이 (10, 75)인 배열로 만들어 준다.

## 풀링 계층 구현하기

```python
class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = (H - self.pool_h) // self.stride + 1
        out_w = (W - self.pool_w) // self.stride + 1

        # 전개 (1)
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        # 최댓값 (2)
        # np.max메서드는 인수로 축(axis)을 지정할 수 있는데, 이 인수로 지정한 축마다 최댓값을 구할 수 있다.
        # 아래와 같은 경우는 입력 xdml 1번째 차원의 축마다 최댓값을 구한다.
        out = np.max(col, axis=1)

        # 성형 (3)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        return out

```

1. 입력 데이터를 im2col로 전개
2. 행별 최댓값을 구함
3. 적절한 모양으로 성형

---

# CNN 구현하기

**[SimpleConvNet]**

```python
class SimpleConvNet:
    def __init__(self, input_dim=(1, 28, 28), conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1}, 
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_szie + 2*filter_pad) / filter_stride+1
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))
```

<aside>
💡

합성곱 계층의 크기를 미리 계산하는 이유?

</aside>

1. **다음 층(특히 완전연결층, FC layer)과 연결하기 위해**

- 완전연결층(FC)은 입력 크기가 **고정된 1차원 벡터**여야 하므로,
    
    합성곱/풀링을 거친 뒤 **몇 개의 뉴런이 나오나(출력 차원)** 를 알아야 `W`(가중치 행렬)를 올바른 크기로 초기화할 수 있다.
    
- 즉, `conv_output_size`와 `pool_output_size`를 계산해야 **다음 층 weight shape**를 정할 수 있다.
- 컨볼루션과 풀링을 거친 후의 출력 데이터 모양은 `(filter_num, pool_output_height, pool_output_width)`가 됩니다.

2. **메모리(파라미터) 초기화에 필요**

- 신경망의 각 층 가중치는 `np.random.randn(...) * std` 이런 식으로 초기화한다.
- 이때 **행렬 크기**가 정확히 맞지 않으면 에러가 난다.
- 그래서 Conv → Pooling 후의 출력 크기를 정확히 알아야,
    
    `W2`(은닉층), `W3`(출력층) 등 FC 계층의 weight를 미리 올바른 차원으로 잡을 수 있다.
    

3. **오류 방지 & 가독성**

- 합성곱 출력 크기 공식:(I: 입력 크기, F: 필터 크기, P: 패딩, S: 스트라이드)
    
    ![image.png](image%201.png)
    
- 이걸 코드 안에서 바로 쓰면 실수하기 쉽다.
- 따라서 **초기화 단계에서 미리 한 번 계산해 변수로 저장**하면, 이후 네트워크 구현과 디버깅이 훨씬 편해진다.

```python
# 매개변수 초기화 코드
        self.params = {}
        # '\'는 아직 줄이 끝나지 않았다는 의미
        self.params['W1'] = weight_init_std * \
                            np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)
```

학습에 필요한 매개변수는 1번째 층의 합성곱 계층과 나머지 두 완전연결 계츠으이 가중치와 편향이다.

이 매개변수들을 인스턴스 변수 params에 저장한다.

1번쨰 층의 합성곱 계층의 가중치를 W1, 편향을 b1이라는 키로 저장하고 2번째 층의 완전연결 계층의 가중치와 편향을 W2, b2 마지막 3번쨰 층의 완전연결 계층의 가중치와 편향을 W3와 b3 키로 각각 저장

```python
# CNN을 구성하는 계층들 생성
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

        self.last_layer = SoftmaxWithLoss()
```

**[predict method] → 추론을 수행**

```python
def predict(self, x):
    for layer in self.layers.values():
        x = layer.forward(x)
    return x

def loss(self, x, t):
    y = self.predict(x)
    return self.last_layer.forward(y, t)
```

x는 입력 데이터, t는 정답 레이블이다. 추론을 수행하는 predic메서드는 초기화 때 layers에 추가한 계층을 맨 앞에서부터 차례로 forward메서드를 호출하며 그 결과를 다음 계층에 전달한다. 손실 함수를 구하는 loss메서드는 preict 메서드의 결과를 인수로 마지막 층의 forward메서드를 호출한다. 즉, 첫 계층부터 마지막 계층까지 forward를 처리한다.

**[오차역전파법] → 기울기 구하기**

```python
def gradient(self, x, t):
    # 순전파
    self.loss(x, t)

    # 역전파
    dout = 1
    dout = self.last_layer.backward(dout)

    layers = list(self.layers.values())
    layers.reverse()
    for layer in layers:
        dout = layer.backward(dout)

    # 결과 저장
    grads = {}
    grads['W1'] = self.layers['Conv1'].dW
    grads['b1'] = self.layers['Conv1'].db
    grads['W2'] = self.layers['Affine1'].dW
    grads['b2'] = self.layers['Affine1'].db
    grads['W3'] = self.layers['Affine2'].dW
    grads['b3'] = self.layers['Affine2'].db

    return grads
```

`grads` 딕셔너리에 기울기를 저장

# CNN 시각화하기

## 1번째 층의 가중치 시각화하기

위에서 MNIST 데이터셋으로 간단한 CNN학습을 했는데, 그때 1번쨰 층의 합성곱 계층의 가중치는 형상이 (30, 1, 5, 5) 즉, (필터 30개, 채널 1개, 5x5크기)이다.

필터가 5x5이고 채널이 1개라는 것은 이 필터를 1채널의 회색조 이미지로 시각화할 수 있다는 뜻이다.

![image.png](image%202.png)

학습전 필터는 무작위로 초기화되고 있어 흑백의 정도에 규칙성이 없다. 하지만, 학습을 마친 필터는 규칙성 있는 이미지가 되었다. 흰색에서 검은색으로 변화하는 필터와 덩어리(블롭blob)가 진 필터 등 규칙을 띄는 필터로 바뀌었다.

오른쪽같이 규칙성 있는 필터는 에지(색상이 바뀐 경계선)과 블롭(국소적으로 덩어리진 영역)등을 보고있다.

## 층 깊이에 따른 추출 정보 변화

1번쨰 층의 합성곱 계층에서는 에지나 블롭 등의 저수준 정보가 추출된다. 반면 겹겹이 쌓인 CNN은 계층이 깊어질수록 추출되는 정보(정확히는 강하게 반응하는 뉴런)는 더 추상화 된다.

![image.png](image%203.png)

이 네트워크 구조는 AlexNet이라 한다. 합성곱 계층과 풀링 계층을 여러 겹 쌓고, 마지막으로 완전연결 계층을 거쳐 결과를 출력하는 구조이다. 층이 깊어지면서 더 복잡하고 추상화된 정보가 추출된다. 처음 층은 단순한 에지에 반응하고, 이어서 텍스처, 그리고 더 복잡한 사물의 일부에 반응하도록 변화한다.

즉, 층이 깊어지면서 뉴런이 반응하는 대상이 단순한 모양에서 ‘고급’정보로 변화해가고 이를 다시 말하면 사물의 ‘의미’를 이해하도록 변화하는 것이다.

# 대표적인 CNN

## LeNet

![image.png](image%204.png)

- 손글씨 숫자를 인식하는 네트워크로 1998년에 제안됨
- 합성곱 계층과 풀링 계층을 반복하고, 마지막으로 완전연결 계층을 거치면서 결과 출력
- LeNet은 시그모이드를 활성화 함수로 사용
- 서브샘플링을 하여 중간 데이터의 크기를 줄이지만 현재는 최대 풀링이 주류임

## AlexNet

![image.png](image%205.png)

- 2012년에 발표됨
- 합성곱 계층과 풀링 계층을 거듭하며 마지막으로 완전연결 계층을 거침
- 활성화 함수를 ReLU로 사용
- LRN(Local Response Normalization)이라는 국소적 정규화 실시하는 계층 이용
- 드롭아웃 사용