#여러 문제가 한 파일에 포함되어 있어서 마지막 문항을 제외한 각 문제마다의 정답은(print 구문) 주석처리 해두었습니다.

import pandas as pd
import numpy as np

df = pd.read_excel("C:/Users/sj5bl/Documents/관서별 5대범죄 발생 및 검거.xlsx")

#1번
#df 데이터프레임의 1,2번 인덱스 컬럼 삭제
df1 = df.drop(df.columns[[1,2]],axis=1)
#df 데이터프레임의 0번 인덱스 row 삭제
df1 = df1.drop([0],axis=0)
# print(df1)

#2번
#매핑할 이름을 딕셔너리 형태로 저장
dic = {'서대문서': '서대문구', '수서서': '강남구', '강서서': '강서구', '서초서': '서초구',
'서부서': '은평구', '중부서': '중구', '종로서': '종로구', '남대문서': '중구',
'혜화서': '종로구', '용산서': '용산구', '성북서': '성북구', '동대문서': '동대문구',
'마포서': '마포구', '영등포서': '영등포구', '성동서': '성동구', '동작서': '동작구',
'광진서': '광진구', '강북서': '강북구', '금천서': '금천구', '중랑서': '중랑구',
'강남서': '강남구', '관악서': '관악구', '강동서': '강동구', '종암서': '성북구', 
'구로서': '구로구', '양천서': '양천구', '송파서': '송파구', '노원서': '노원구', 
'방배서': '서초구', '은평서': '은평구', '도봉서': '도봉구'}

df2 = df
#특정 컬럼의 요소값(스칼라값)만을 수정하여서 apply 대신 map 사용
#관서명이 dic에 있으면 매핑된 결과값(dic[x])을 반환하고, 없으면 NaN값 반환
df2['구별'] = df2['관서명'].map(lambda x: dic[x] if x in dic else np.nan)
#'구별' 컬럼의 NaN 값을 '구 없음' 으로 대체
df2['구별'] = df2['구별'].fillna('구 없음')
# print(df2)

#3번
#'구별' 컬럼을 인덱스로 하고, 동일한 인덱스의 값은 sum으로 연산하는 피벗테이블 생성
df = df.pivot_table(index="구별", aggfunc="sum")
df.drop(columns='관서명', inplace=True)
# print(df)

#4번
df.drop(['구 없음'], inplace=True)

#5번
#axis=1을 사용해 기존 행간의 연산을 컬럼간 연산으로 사용
df['강간검거율'] = df.apply(lambda row : 100*row['강간(검거)']/row['강간(발생)'], axis=1)
df['살인검거율'] = df.apply(lambda row : 100*row['살인(검거)']/row['살인(발생)'], axis=1)
df['절도검거율'] = df.apply(lambda row : 100*row['절도(검거)']/row['절도(발생)'], axis=1)
df['폭력검거율'] = df.apply(lambda row : 100*row['폭력(검거)']/row['폭력(발생)'], axis=1)
df['검거율'] = df.apply(lambda row : 100*row['소계(검거)']/row['소계(발생)'], axis=1)
# print(df)

#6번
del df['강간(검거)']
del df['강도(검거)']
del df['살인(검거)']
del df['절도(검거)']
del df['폭력(검거)']
del df['소계(발생)']
del df['소계(검거)']
# print(df)

#7번
dic7 = {'강간(발생)':'강간',
'강도(발생)':'강도',
'살인(발생)':'살인',
'절도(발생)':'절도',
'폭력(발생)':'폭력'}
df.rename(columns=dic7, inplace=True)
# print(df)

#도전과제 1번
df_add = pd.read_csv("C:/Users/sj5bl/Documents/pop_kor.csv")
df_add.set_index('구별',inplace = True)
# print(df_add)

#도전과제 2번
df_add = df_add.join(df)
# print(df_add)

#도전과제 3번
df_add.sort_values(by='검거율')
print(df_add)