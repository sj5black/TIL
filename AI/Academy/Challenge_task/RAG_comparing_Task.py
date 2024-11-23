import sqlite3
import os
import json

# SQLite 데이터베이스 연결
conn = sqlite3.connect("AI_news_crolling.db")
cursor = conn.cursor()

# 테이블 생성
cursor.execute("""
CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT, -- 자동 증가하는 기본 키
    title TEXT NOT NULL,
    description TEXT,
    url TEXT,
    date TEXT,
    content TEXT
)
""")

# JSON 데이터 로드
json_folder = "./AI_TIMES_Crollingfiles/"

for filename in os.listdir(json_folder):
    if filename.endswith(".json"):
        with open(os.path.join(json_folder, filename), "r", encoding="utf-8") as file:
            data = json.load(file)

            # JSON 데이터를 SQLite DB에 삽입
            for item in data:
                cursor.execute("""
                INSERT INTO articles (title, description, url, date, content)
                VALUES (?, ?, ?, ?, ?)
                """, (item["title"], item["description"], item["url"], item["date"], item["content"]))

# content 필드 읽기
cursor.execute("SELECT content FROM articles")
rows = cursor.fetchall()

# 데이터 확인
for row in rows[:5]:  # 상위 5개 데이터 확인
    print(row)  # 길이 제한

# 변경 사항 저장/DB 종료
conn.commit()
conn.close()


from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

def summarize_content(content):
    """GPT를 사용해 주어진 content를 요약"""
    try:
        response = llm.predict_messages(
            messages=[
                {"role": "system", "content": "당신은 뛰어난 요약 전문가입니다."},
                {"role": "user", "content": f"다음 글을 간결하게 요약해 주세요:\n\n{content}"}
            ],
            max_tokens=200
        )
        summary = response.content.strip()
        return summary
    except Exception as e:
        print(f"요약 중 오류 발생: {e}")
        return None

# 요약정보 담기 (100개 내용까지만 요약)
summary_list=[]
for i, row in enumerate(rows):
    summary_list.append(summarize_content(row))
    print(f"{i}번째 완료")
    if i >=100 : break
print(summary_list[1])


from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pprint import pprint

#요약되지 않은 원본정보 담기
original_list = [row[0] for row in rows[:100]]

# 2가지 버전의 벡터스토어 생성 (원본/요약)
vectorstore_summary = Chroma.from_texts(texts=summary_list, embedding=OpenAIEmbeddings())
vectorstore_original = Chroma.from_texts(texts=original_list, embedding=OpenAIEmbeddings())

# 2가지 버전의 리트리버 생성 (원본/요약)
retriever_s = vectorstore_summary.as_retriever()
retriever_o = vectorstore_original.as_retriever()

# 리트리버를 참조하는 시스템 프롬프트 생성
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question briefly, based on the following context. \n\n {context}")
])

# RAG 체인 선언 (원본/요약)
rag_chain_s = (
        {"context": retriever_s}
        | prompt
        | llm
        | StrOutputParser()
)

rag_chain_o = (
        {"context": retriever_o}
        | prompt
        | llm
        | StrOutputParser()
)

# 챗봇 대화 시작
query=""
while query!="exit": 
    query = str(input("Chat to GPT : "))
    
    # GPT 답변 출력
    answer_o = rag_chain_o.invoke(query)
    print(" - GPT Answer based on original - ")
    pprint(answer_o)
    answer_s = rag_chain_s.invoke(query)
    print(" - GPT Answer based on summary - ")
    pprint(answer_s)
    print("\n")