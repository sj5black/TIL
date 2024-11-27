import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 예시 데이터프레임 생성
data = {
    '날짜': ['2023-01-01', '2023-01-02', '2023-01-01', '2023-01-02'],
    '도시': ['서울', '서울', '부산', '부산'],
    '온도': [2, 3, 6, 7],
    '습도': [55, 60, 80, 85]
}
df = pd.DataFrame(data)

print(df)