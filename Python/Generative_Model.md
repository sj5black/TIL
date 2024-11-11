### 생성형 모델 (Generative Model)
- 주어진 데이터를 바탕으로 새로운 데이터를 생성하는 AI 모델

- **텍스트 생성**
GPT-3, ChatGPT 등이 대표적인 텍스트 생성 모델로, 사용자가 제공한 몇 개의 단어 또는 문장을 기반으로 새로운 텍스트를 생성

- **이미지 생성**
DALL-E, Stable Diffusion 같은 모델은 텍스트 설명을 바탕으로 새로운 이미지를 생성

- **오디오 생성**
Jukedeck, OpenAI의 Jukebox는 주어진 멜로디나 텍스트를 기반으로 음악을 작곡하거나 오디오를 생성

### 랜덤성(Randomness)과 조건성(Conditionality)

**랜덤성(Randomness)**
 - 생성형 모델이 다양한 결과를 생성할 수 있도록 도와주는 요소

**확률 분포**
- 학습 데이터를 통해 얻은 확률 분포를 기반으로 새로운 데이터를 생성
- 예를 들어, 텍스트 생성 모델은 다음에 올 단어를 예측할 때 각 단어의 확률을 계산하고, 그 확률에 따라 랜덤하게 단어를 선택

**조건성(Conditionality)**
- 생성형 모델이 특정 조건을 기반으로 데이터를 생성하는 능력

```py
import os

os.environ["OPENAI_API_KEY"] = "<your OpenAI API key>"

from openai import OpenAI
client = OpenAI()

response = client.images.generate(
  model="dall-e-3",
  prompt="a white siamese cat",
  size="1024x1024",
  quality="standard",
  n=1,
)

image_url = response.data[0].url
```
---
## 텍스트 기반 생성형 모델의 원리

텍스트 기반 생성형 AI는 주어진 텍스트 입력(조건)을 바탕으로 새로운 텍스트를 생성

 **1. 입력 토큰화**
사용자가 입력한 텍스트를 토큰(단어 또는 서브워드)으로 변환

**2. 확률 예측**
모델은 주어진 텍스트를 기반으로 다음에 올 단어의 확률을 예측

**3. 랜덤 선택**
예측된 확률 분포에서 랜덤하게 다음 단어를 선택. 이때 **temperature** 파라미터를 조정하여 랜덤성을 조절

**4. 반복 생성**
이러한 과정을 반복하여 문장이 완성될 때까지 텍스트를 생성

## 이미지 기반 생성형 모델의 원리

**1. 텍스트 인코딩**
입력된 텍스트 조건을 벡터로 인코딩하여 모델에 입력

**2. 이미지 생성**
모델은 인코딩된 텍스트와 함께 이미지의 주요 특징(예: 형태, 색상, 질감 등)을 생성

**3. 세부 사항 추가**
랜덤성을 적용하여 세부적인 이미지 요소를 생성하고, 이를 합성하여 최종 이미지를 생성

## 오디오 기반 생성형 모델의 원리

**1. 텍스트 인코딩**
입력된 텍스트(예: 노래 가사)나 멜로디를 인코딩하여 모델에 입력

**2. 오디오 생성**
인코딩된 입력을 바탕으로 오디오 신호를 생성 -> 이 과정에서 음색, 리듬, 멜로디 등을 조합

**3. 랜덤성 적용**
랜덤성을 통해 음성의 미세한 변화를 추가하여, 동일한 조건에서도 다양한 오디오를 생성