from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import streamlit as st
# from RAG_comparing_Task import AI_custom_model  # 함수 가져오기

# Streamlit UI 구성
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")

st.title("AI Chatbot")
st.write("GPT와 대화하고, 요약 및 원본 데이터를 기반으로 답변을 비교하세요.")

# 대화 상태 관리
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 벡터스토어 로드
try:
    vectorstore_summary = Chroma(
        persist_directory="./chroma_summary_store",
        embedding_function=OpenAIEmbeddings()
    )
    vectorstore_original = Chroma(
        persist_directory="./chroma_original_store",
        embedding_function=OpenAIEmbeddings()
    )

    retriever_s = vectorstore_summary.as_retriever()
    retriever_o = vectorstore_original.as_retriever()

except Exception as e:
    print(f"벡터스토어 로드 중 오류 발생: {e}")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pprint import pprint

llm = ChatOpenAI(model="gpt-4o-mini")

# 챗봇 대화 시작
def AI_custom_model(query : str):
    
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

    if query == "exit" : return "대화를 종료합니다."

    # GPT 답변 출력
    answer_o = rag_chain_o.invoke(query)
    print(" - GPT Answer based on original - ")
    pprint(answer_o)
    answer_s = rag_chain_s.invoke(query)
    print(" - GPT Answer based on summary - ")
    pprint(answer_s)
    print("\n")

    answer = f" - GPT Answer based on original - \n{answer_o}\n\n - GPT Answer based on summary - \n{answer_s}"
    return answer

# 사용자 입력
user_input = st.text_input("메시지를 입력하세요:", key="user_input")
if st.button("전송") and user_input:
    # GPT 응답 생성
    with st.spinner("AI 응답 생성 중..."):
        try:
            result = AI_custom_model(user_input)  # 함수 호출
            st.session_state["messages"].append({"role": "user", "content": user_input})
            st.session_state["messages"].append({"role": "bot", "content": result})
        except Exception as e:
            st.error(f"응답 생성 중 오류 발생: {e}")

# 대화 기록 표시
for message in st.session_state["messages"]:
    if message["role"] == "user":
        st.text_area("👤 사용자:", value=message["content"], height=100)
    elif message["role"] == "bot":
        st.text_area("🤖 AI:", value=message["content"], height=230)

