import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
from typing import List, Dict

def get_hollys_store_info(page: int) -> str:
    """
    할리스 매장 정보 페이지의 HTML을 가져오는 함수
    """
    url = "https://www.hollys.co.kr/store/korea/korStore2.do"
    params = {"pageNo": page}
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0'
    }
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"페이지 {page} 요청 중 에러 발생: {e}")
        return ""

def parse_store_info(html: str) -> List[Dict]:
    """
    HTML에서 매장 정보를 파싱하는 함수
    """
    stores = []
    soup = BeautifulSoup(html, 'html.parser')
    
    tbody = soup.find("tbody")
    if not tbody:
        return stores
        
    for tr in tbody.find_all("tr"):
        tds = tr.find_all('td')
        if len(tds) < 6:
            continue
            
        store = {
            'region': tds[0].text.strip(),
            'name': tds[1].text.strip(),
            'status': tds[2].text.strip(),
            'address': tds[3].text.strip(),
            'service': ' '.join(img['alt'] for img in tds[4].find_all('img')),
            'phone': tds[5].text.strip()
        }
        stores.append(store)
        
    return stores

def save_to_files(stores: List[Dict], base_path: str = "./data"):
    """
    매장 정보를 CSV와 JSON 파일로 저장하는 함수
    """
    # CSV 파일로 저장 (pandas 사용)
    df = pd.DataFrame(stores)
    print(df)
    df.to_csv(f"{base_path}/hollys_stores.csv", encoding='utf-8', index=False)
    
    # JSON 파일로 저장
    with open(f"{base_path}/hollys_stores.json", 'w', encoding='utf-8') as f:
        json.dump(stores, f, ensure_ascii=False, indent=4)

def main():
    stores = []
    
    # 매장 정보 수집
    print("할리스 매장 정보 수집 중...")
    for page in range(1, 3):  # 1~2 페이지
        html = get_hollys_store_info(page)
        if html:
            stores.extend(parse_store_info(html))
            print(f"페이지 {page} 완료")
    
    if not stores:
        print("매장 정보를 가져오는데 실패했습니다.")
        return
        
    print(f"\n총 {len(stores)}개의 매장 정보를 수집했습니다.")

    # 데이터 저장
    try:
        save_to_files(stores, './')
        print("데이터 저장 완료!")
        print("- hollys_stores.csv")
        print("- hollys_stores.json")
        
        # 데이터 미리보기
        df = pd.DataFrame(stores)
        print("\n데이터 미리보기:")
        print(df.head())
        
    except Exception as e:
        print(f"데이터 저장 중 에러 발생: {e}")

if __name__ == "__main__":
    main()