### BERT의 사전 학습 단계

1. Masked Language Modeling (MLM)
문장의 일부 단어를 마스킹(masking)한 후, 이를 예측하도록 모델을 학습 -> 문맥을 양방향으로 이해

2. Next Sentence Prediction (NSP)
두 문장이 주어졌을 때, 두 번째 문장이 첫 번째 문장 뒤에 자연스럽게
이어지는지 예측 -> 문장 간의 관계를 이해하는 능력 학습

```py
#1 BERT를 그대로 학습

from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score

dataset = load_dataset("imdb")
test_dataset = dataset["test"].shuffle(seed=42).select(range(500))
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding = "max_length", truncation=True)

test_dataset = test_dataset.map(tokenize_function, batched=True)
test_dataset.set_format(type="torch", columns = ['input_ids', 'attention_mask','label'])

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

model.eval()

all_prediction = []
all_labels =[]

for batch in torch.utils.data.DataLoader(test_dataset, batch_size=8):
    with torch.no_grad():
        outputs = model(input_ids = batch['input_ids'], attention_mask=batch['attention_mask'])
        logits = outputs.logits
        preds = np.argmax(logits.numpy(), axis=1)
        all_prediction.extend(preds)
        all_labels.extend(batch['label'].numpy())

        
# 정확도 계산 (accuracy test)
acc = accuracy_score(all_labels, all_prediction)
print(f"Accuracy without fine-tuning : {acc:.4f}")

# Accuracy without fine-tuning : 0.5080
```

```py
#2 BERT 를 파인튜닝하여 모델 학습
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# IMDb 데이터셋 로드
dataset = load_dataset("imdb")

# 훈련 및 테스트 데이터셋 분리
train_dataset = dataset['train'].shuffle(seed=42).select(range(1000))  # 1000개 샘플로 축소
test_dataset = dataset['test'].shuffle(seed=42).select(range(500))  # 500개 샘플로 축소

# BERT 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 데이터셋 토크나이징 함수 정의
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

# 데이터셋 토크나이징 적용
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# 모델 입력으로 사용하기 위해 데이터셋 포맷 설정
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# BERT 모델 로드
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

### 파인 튜닝 (fine tuning) ###
# 훈련 인자 설정
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_steps=10_000,
    save_total_limit=2,
)

# 트레이너 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# 모델 훈련
trainer.train()
trainer.evaluate()

import numpy as np
from sklearn.metrics import accuracy_score

# 평가 지표 함수 정의
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)  # 예측된 클래스
    labels = p.label_ids  # 실제 레이블
    acc = accuracy_score(labels, preds)  # 정확도 계산
    return {'accuracy': acc}

# 이미 훈련된 트레이너에 compute_metrics를 추가하여 평가
trainer.compute_metrics = compute_metrics

# 모델 평가 및 정확도 확인
eval_result = trainer.evaluate()
print(f"Accuracy: {eval_result['eval_accuracy']:.4f}")

# Accuracy: 0.9080
```

