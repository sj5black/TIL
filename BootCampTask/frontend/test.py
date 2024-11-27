# 전체 열의 모든 요소값을 기준으로 중복 검사
df.duplicated()

# 'name' 열만을 기준으로 중복 검사
df.duplicated(subset=['name'])


#해당 행에 있는 각 컬럼값들 중 NaN 인 것들의 합 계산
df_products['missing count'] = df_products.isnull().sum(axis=1)
