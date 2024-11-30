import requests
import random

# 1. 데이터 생성
ecg_data = [round(random.uniform(8000, 10000), 1) for _ in range(15000)]  # 15000개의 랜덤 데이터

# 2. POST 요청 데이터 준비
#url = "http://192.168.34.31:8081/ws/ecg"  # 요청을 보낼 URL
#url = "http://192.168.34.40:8081/ws/ecg"   #규혁이 주소  l7562l@naver.com
#url = "http://127.0.0.1:8082/predict"  #fastapi 서버 주소 
url = "https://reptile-promoted-publicly.ngrok-free.app/ws/ecg"
# "device_id": "d8:3a:dd:2f:35:94",

payload = {
    "device_id": "d8:3a:dd:2f:35:94",
    "userid": "ww",
    "ecgdata": ecg_data
}
headers = {
    "Content-Type": "application/json"
}

# 3. POST 요청 보내기
try:
    response = requests.post(url, json=payload, headers=headers)
    # 4. 응답 출력
    print("Status Code:", response.status_code)
    print("Response Body:", response.text)
except requests.exceptions.RequestException as e:
    print("Error occurred:", e)