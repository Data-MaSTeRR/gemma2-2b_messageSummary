import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ollama  # Gemma 모델 사용
from fastapi.responses import JSONResponse
import re

app = FastAPI(debug=True)

# Ollama 서버 URL 설정 (환경 변수로 가져옴)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama-server:11434")

# Ollama 클라이언트 초기화
ollama_client = ollama.Client(host=OLLAMA_URL)

# 요청 데이터 모델 정의
class TextInput(BaseModel):
    text: str

# 요약 결과에서 불필요한 특수문자 제거 및 문장 부호 정리
def clean_summary_text(summary):
    clean_text = re.sub(r'[^\w\sㄱ-ㅎ가-힣.,!?]', '', summary)  # 한국어, 영문자, 숫자 및 . , ! ? 만 남김
    clean_text = re.sub(r'\s+', ' ', clean_text)  # 여러 공백을 하나의 공백으로 변환
    return clean_text.strip()

@app.post("/summarize_and_classify")
async def summarize_and_classify(input_data: TextInput):
    text = input_data.text  # 입력 텍스트

    # 1. 요약 기능 (Gemma 모델 사용)
    try:
        summarize_prompt = f"""
        
        만약 classification이 일반이라면, 최대 2문장으로 요약해.
        만약 classification이 광고/스팸이라면, 앞에 광고/스팸이라고 알려주고, 왜 그런지 한 문장으로 설명해줘.

        문자메시지: {text}
        요약 (친절한 안내문 스타일, "습니다" 체로, 최대 2문장):
        """
        summary_response = ollama_client.generate(model="gemma2:2b", prompt=summarize_prompt)
        clean_summary = re.sub(r'[\n\r]+', ' ', summary_response['response']).strip()  # 요약 결과 처리
        clean_summary = clean_summary_text(clean_summary)  # 불필요한 특수문자 제거 및 정리
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"요약 중 오류 발생: {str(e)}")

    # 2. 광고/스팸 분류 (Gemma 모델 사용)
    try:
        classify_prompt = f"""
        주어진 텍스트가 광고 또는 스팸인지 확인해 주세요. 
        광고나 스팸이면 "광고/스팸"이라고 대답해 주세요. 
        일반적인 메시지라면 "일반"이라고 대답해 주세요.

        텍스트: {text}
        결과:
        """
        classify_response = ollama_client.generate(model="gemma2:2b", prompt=classify_prompt)
        classification_result = classify_response['response'].strip()  # 분류 결과 처리
        
        # 결과가 예상과 다를 경우 강제로 설정
        if classification_result not in ["광고/스팸", "일반"]:
            print("예상치 못한 분류 결과:", classification_result)  # 디버그 용도
            classification_result = "일반"  # 기본적으로 일반으로 설정
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"광고/스팸 분류 중 오류 발생: {str(e)}")
    
    # 최종 JSON 응답 반환
    return JSONResponse(content={"summary": clean_summary, "classification": classification_result}, media_type="application/json; charset=utf-8")
