#pip install requests

import requests

def get_related_searches(search_term):
    """
    네이버 연관검색어를 가져오는 함수
    Args:
        search_term (str): 검색하고자 하는 단어
    Returns:
        list: 연관검색어 리스트
    """

    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    headers =  {
        'User-Agent': user_agent
    }

    # 네이버 자동완성 API URL
    base_url = "https://ac.search.naver.com/nx/ac"
    
    # 검색어를 URL 인코딩하여 파라미터 구성
    params = {
        'q': search_term,
        'st': 100
    }
    
    try:
        # API 요청
        response = requests.get(base_url, params=params, headers=headers)

        # JSON 응답 파싱
        data = response.json()
        print(data)
        # 연관검색어 추출 (items[0]의 각 항목의 첫번째 요소)
        related_searches = [item[0] for item in data['items'][0]]
        return related_searches
        
    except requests.exceptions.RequestException as e:
        print(f"에러 발생: {e}")
        return []

def main():
    print("<< 네이버 연관 검색어 가져오기 >>")
    search_term = "파이썬"
    
    # 연관검색어 가져오기
    related_terms = get_related_searches(search_term)
    
    # 결과 출력
    if related_terms:
        print("\n연관검색어:")
        for i, term in enumerate(related_terms, 1):
            print(f"{i}. {term}")
    else:
        print("연관검색어를 가져오는데 실패했습니다.")

if __name__ == "__main__":
    main()