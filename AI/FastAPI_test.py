from fastapi import FastAPI

app = FastAPI()

# 터미널 창에서 uvicorn 파일이름:app --reload 시 서버 실행
# http://127.0.0.1:8000/docs 에서 해당 웹페이지의 경로들 확인 가능 (/test 등..)

@app.get("/") # 도메인의 루트 경로 (기본 경로 http://127.0.0.1:8000/ 에서 실행)
def read_root():
    return {"message": "Hello World!"}

@app.get("/test") #  http://127.0.0.1:8000/test 경로에서 실행할 파일
def read_root():
    return {"message": "This is test Message."}