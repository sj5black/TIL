from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from rag_model import get_question, get_feedback, get_session_no  # rag_model.py 파일을 임포트
import uvicorn
import logging

# 모델 메서드 호출에 필요한 변수 선언
user_id: str = ""
session_no: int = get_session_no(user_id)
type_: str = ""
order: int = 0

# FastAPI 애플리케이션 생성
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 보안상 필요한 도메인만 추가
    allow_credentials=True,
    allow_methods=["GET", "POST"], # GET, POST 요청만 허용
    allow_headers=["*"],
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API 요청 모델
class QuizRequest(BaseModel):
    topic: str

    @validator("topic")
    def validate_topic(cls, value):
        if not value.strip():
            raise ValueError("Topic must not be empty.")
        return value

class AnswerRequest(BaseModel):
    context: str
    answer: str

    @validator("context", "answer")
    def validate_not_empty(cls, value):
        if not value.strip():
            raise ValueError("Fields must not be empty.")
        return value



# 퀴즈 생성 엔드포인트
@app.post("/generate_quiz")
async def generate_quiz(request: QuizRequest):
    try:
        logger.info(f"Generating quiz for topic: {request.topic}")
        quiz = get_question(request.topic)
        return {"quiz": quiz}
    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

# 답변 검토 엔드포인트
@app.post("/check_answer")
async def check_answer(request: AnswerRequest):
    try:
        logger.info(f"Checking answer for context: {request.context}")
        result = create_answer(request.context, request.answer)
        return {"result": result}
    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

# 메인 함수
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
