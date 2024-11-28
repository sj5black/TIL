from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pprint import pprint

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re

from dotenv import load_dotenv
import os

def Set_Selenium():
    # Selenium 옵션 설정 (헤드리스 모드로 실행)
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # 브라우저 창을 띄우지 않음
    chrome_options.add_argument("--disable-gpu")  # GPU 비활성화 (일부 환경에서 필요)

    # WebDriver 경로 설정 (자동 설치)
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    return driver

def Reference_Load(path : str):
    
    # .env 파일에서 환경변수 로드
    load_dotenv(path) # "C:/.env"
    
    url_list=[]

    # 환경변수에 저장된 URL 로드
    for i in range(1, 100):  # URL_1 ~ URL_16
        
        # 환경변수가 존재하면 추가
        url = os.getenv(f"PYTHON_LIBRARY_URL_{i}")
        if url : url_list.append(url)

        url = os.getenv(f"ML_URL_{i}")
        if url : url_list.append(url)
        
        url = os.getenv(f"DL_URL_{i}")
        if url : url_list.append(url)

        url = os.getenv(f"LLM_RAG_URL_{i}")
        if url : url_list.append(url)

    return url_list


def Crolling(driver, url_list):    
    txt_list=[]
    
    # 웹페이지 요청
    for url in url_list:
        driver.get(url)  # 페이지 로드

        # 특정 요소가 로드될 때까지 기다림 (예: Notion 페이지에서 주요 콘텐츠가 담길 요소)
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".notion-page-content"))
            )
        except TimeoutException:
            print(f"페이지 로딩 실패: {url}")
            continue
        
        # 토글이 닫혀 있으면 토글을 열기
        try:
            # 모든 토글 버튼을 찾음 (Ctrl+Alt+T에 해당하는 토글을 찾아서 열기)
            toggle_buttons = driver.find_elements(By.XPATH, "//div[@role='button' and (@aria-expanded='false')]")

            # 각 토글을 클릭하여 열기
            for button in toggle_buttons:
                driver.execute_script("arguments[0].click();", button)
                time.sleep(1)  # 토글이 열리기 전에 잠깐 대기

        except Exception as e:
            print(f"토글을 여는 데 실패했습니다: {e}")

        # 페이지의 HTML 가져오기
        html_code = driver.page_source

        # BeautifulSoup으로 HTML 파싱
        soup = BeautifulSoup(html_code, 'html.parser')

        txt = soup.get_text()

        # 1. \xa0를 공백으로 변환, 불필요한 문자 제거
        txt = txt.replace('\xa0', ' ')
        
        # txt = txt.replace('모든 토글을 열고 닫는 단축키', '')
        # txt = txt.replace('Windows : Ctrl + alt + t', '')
        # txt = txt.replace('Mac : ⌘ + ⌥ + t', '')
        # txt = txt.replace('Copyright ⓒ TeamSparta All rights reserved', '')

        # 2. 정규식을 사용해 \\로 시작하는 LaTeX 명령어 제거
        txt = re.sub(r'\\[a-zA-Z]+\{.*?\}', '', txt)  # \command{...} 형식 제거
        txt = re.sub(r'\\[a-zA-Z]+', '', txt)        # \command 형식 제거

        # 3. 불필요한 문구 제거
        # txt = re.sub(r'모든 토글을 열고 닫는 단축키', '', txt)
        # txt = re.sub(r'Windows\s*:\s*Ctrl\s*\+\s*alt\s*\+\s*t', '', txt)
        # txt = re.sub(r'Mac\s*:\s*⌘\s*\+\s*⌥\s*\+\s*t', '', txt)
        # txt = re.sub(r'Copyright ⓒ TeamSparta All rights reserved', '', txt)
        # txt = re.sub(r'\s+', ' ', txt).strip()

        # 텍스트만 가져오기
        txt_list.append(txt)

    driver.quit()  # 브라우저 종료

    for i in range(len(txt_list)):
        txt_list[i] = txt_list[i].replace('모든 토글을 열고 닫는 단축키', '')
        txt_list[i] = txt_list[i].replace('Windows : Ctrl + alt + t', '')
        txt_list[i] = txt_list[i].replace('Mac : ⌘ + ⌥ + t', '')
        txt_list[i] = txt_list[i].replace('Copyright ⓒ TeamSparta All rights reserved', '')

    # for txt in txt_list:
    #     txt = txt.replace('모든 토글을 열고 닫는 단축키', '')
    #     txt = txt.replace('Windows : Ctrl + alt + t', '')
    #     txt = txt.replace('Mac : ⌘ + ⌥ + t', '')
    #     txt = txt.replace('Copyright ⓒ TeamSparta All rights reserved', '')

    return txt_list

    