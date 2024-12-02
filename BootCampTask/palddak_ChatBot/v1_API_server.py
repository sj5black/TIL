from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from rag_model import get_feedback, get_session_no, get_question_language, read_quiz_from_file
from rag_model import get_question_language_test, get_feedback_test
from fastapi.responses import JSONResponse
import os
import glob
import uvicorn
import logging
from datetime import datetime

# Audio model libraries
from audio_model import generate_audio_from_text
import io
from pydub import AudioSegment
from fastapi.responses import StreamingResponse

############ 로그 파일 생성 ######################
logging.basicConfig(
    filename="API_server.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("API Server started.")
logger = logging.getLogger(__name__)
##################################################

# 모델 메서드 호출에 필요한 전역변수 선언
global user_id
global session_no
global type_
global order
global quiz
global quiz_list
global current_index
global rag_output_path
global language


user_id: str = "None"
session_no: int = get_session_no(user_id) + 1
type_: str = "python"
order: int = 1
current_index: int = 0
language: str = "한국어"

# FastAPI 애플리케이션 생성
app = FastAPI()
CHATLOG_SERVER_DIR = "./user_chatlog_server"
CHATLOG_CLIENT_DIR = "./user_chatlog_client"
RAG_OUTPUT = "./rag_model_output"

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 보안상 필요한 도메인만 추가
    allow_credentials=True,
    allow_methods=["*"], # GET, POST 요청만 허용
    allow_headers=["*"],
)

###############################################################
###################### 클래스 선언 ############################
###############################################################

# API 요청 모델
class QuizRequest(BaseModel):
    topic: str
    
    #@validator("topic")
    #def validate_topic(cls, value):
    #    if not value.strip():
    #        raise ValueError("Topic must not be empty.")
    #    return value
    pass

class AnswerRequest(BaseModel):
    quiz: str
    user_answer: str
    # @validator("context", "answer")
    # def validate_not_empty(cls, value):
    #     if not value.strip():
    #         raise ValueError("Fields must not be empty.")
    #     return value

class SetUserID(BaseModel):
    requested_user_id: str

class SetBigTopic(BaseModel):
    big_topic: str

class SetSmallTopic(BaseModel):
    small_topic_order: int

class SetLanguage(BaseModel):
    lang: str

class Conversation(BaseModel):
    requested_user_id: str
    chatlog: str

# 텍스트 데이터 모델 정의
class TextRequest(BaseModel):
    text: str

# 서버 로드 체크
@app.get("/server_check")
async def server_check():
    return {"status": "ok"}

# user_id 변경요청 처리
@app.post("/set_user_id")
async def set_type(request: SetUserID):
    user_id = request.requested_user_id  # 요청받은 type을 전역변수 type_에 저장 (str)
    logger.info(f"set_user_id -> {user_id}")
    return {"message": f"Server user_id has been set to: {user_id}"}

# 대주제(type_) 변경요청 처리
@app.post("/set_big_topic")
async def set_type(request: SetBigTopic):
    type_ = request.big_topic  # 요청받은 type을 전역변수 type_에 저장 (str)
    logger.info(f"set_big_topic -> type_ : {type_}")
    return {"message": f"Selected type has been set to: {type_}"}

# 소주제(order) 변경요청 처리
@app.post("/set_small_topic")
async def set_type(request: SetSmallTopic):
    order = request.small_topic_order  # 요청받은 order값을 전역변수 order에 저장 (int)
    logger.info(f"set_small_topic -> order : {order}")
    return {"message": f"Selected type has been set to: {order}"}

# 퀴즈 생성 엔드포인트
@app.post("/generate_quiz")
async def generate_quiz(request: QuizRequest):
    logger.info(f"generate_quiz initial type_ : {type_}")
    logger.info(f"generate_quiz initial session_no : {session_no}")
    logger.info(f"generate_quiz initial user_id : {user_id}")
    logger.info(f"generate_quiz initial request.topic : {request.topic}")
    logger.info(f"generate_quiz initial order : {order}")

    # if type_ != request.topic:
    #     raise HTTPException(status_code=400, detail="선택된 주제와 AI가 생성한 topic이 일치하지 않았습니다.")
    try:
        logger.info(f"Generating quiz for topic: {request.topic}")
        quiz = get_question_language(session_no, user_id, request.topic, order, language, RAG_OUTPUT, current_index)
        return {"QUIZ": quiz}
    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

# 답변 검토 엔드포인트
@app.post("/check_answer")
async def check_answer(request: AnswerRequest):

    # quiz = read_quiz_from_file(RAG_OUTPUT, "quiz", user_id, session_no, type_, order)
    # logger.info(f"QUIZ : {quiz}")
    try:
        logger.info(f"Checking answer for context {request.user_answer}")
        feedback = get_feedback(session_no, user_id, type_, order, request.quiz, request.user_answer, language, RAG_OUTPUT)
        return {"FeedBack": feedback}
    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

##################### 도전 과제 #########################

# 언어 변경요청 처리
@app.post("/set_language")
async def set_type(request: SetLanguage):
    language = request.lang  # 요청받은 lang값을 전역변수 language에 저장 (str)
    logger.info(f"set_language -> order : {order}")
    return {"message": f"Selected type has been set to: {order}"}

# 대화 불러오기 api
@app.get("/get_history/{user_id}")
async def get_history(user_id: str):
    # 파일 경로 패턴
    pattern = os.path.join(CHATLOG_SERVER_DIR, f"{user_id}_*.txt")
    
    # 패턴에 맞는 파일경로들을 str 형식으로 file_paths 리스트에 저장
    file_paths = glob.glob(pattern)
    print(file_paths)

    if not file_paths:
        return []
    
    files_content = []
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            files_content.append({
                "file_name": os.path.basename(file_path), # 파일 이름(경로상의 마지막 이름)만 추출
                "content": content
            })
    return JSONResponse(content=files_content)

# 대화 저장 API
@app.post("/save_conversation")
async def save_conversation(conversation: Conversation):
    # 현재 시간을 밀리초 단위로 포함하여 파일 이름 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # 예: 20241126_153045_123456
    file_name = f"{conversation.requested_user_id}_{timestamp}.txt"
    file_path = os.path.join(CHATLOG_SERVER_DIR, file_name)
    
    # 파일 디렉토리가 없으면 생성
    if not os.path.exists(CHATLOG_SERVER_DIR):
        os.makedirs(CHATLOG_SERVER_DIR)
    
    # 대화 내용 파일에 추가
    with open(file_path, "a", encoding="utf-8") as file:
        file.write(conversation.chatlog)
    
    return {"message": "Conversation saved successfully."}

# 음성파일 요청 처리
@app.post("/generate_audio/")
async def generate_audio_endpoint(text_request: TextRequest):
    audio_content = generate_audio_from_text(text_request.text)

    if audio_content:
        # 음성을 파일로 변환하지 않고 바로 스트리밍
        audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_content))

        # api가 잘 작동하는지 보기위한 저장
        # output_filename = "generated_audio.mp3"
        # audio_segment.export(output_filename, format="mp3")
        # print(f"Audio saved as {output_filename}")

        audio_io = io.BytesIO()
        audio_segment.export(audio_io, format="mp3")
        audio_io.seek(0)
        
        return StreamingResponse(audio_io, media_type="audio/mpeg")
    else:
        raise HTTPException(status_code=500, detail="Failed to generate audio")

# 메인 함수
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# 아이디 서버에 넘기는 부분 추가 필요