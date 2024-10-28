Padding : 자연어로 이루어진 리스트에서 텐서 변환을 위해 빈 칸을 0이나 <PAD> 로 채워주는 처리
Collate funciton : DataLoader에서 배치 데이터를 생성할 때 각 샘플을 결합하는 방법을 정의하는 함수
```python
from torch.utils.data import DataLoader, Dataset

class SimpleDataset(Dataset):
    def __init__(self):
        self.data = [
            torch.tensor([1, 2, 3]),   # 길이 3
            torch.tensor([4, 5]),      # 길이 2
            torch.tensor([6, 7, 8, 9]) # 길이 4
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    # 배치에서 시퀀스 추출
    return pad_sequence(batch, batch_first=True)

# 데이터셋과 DataLoader 생성
dataset = SimpleDataset()
dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

# 데이터 로드
for batch in dataloader:
    print(batch)

------------출력----------------------
tensor([[1, 2, 3],
        [4, 5, 0]])

tensor([[6, 7, 8, 9]])
```
