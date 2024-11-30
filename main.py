from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# 모델 및 스케일러 로드
ECG_MODEL_PATH = "C:\\Users\\sunmoon\\IdeaProjects\\IOT_Back_Server\\model\\ECG_model.pkl"
ECG_SCALER_PATH = "C:\\Users\\sunmoon\\IdeaProjects\\IOT_Back_Server\\model\\ECG_scaler.pkl"
EMG_MODEL_PATH = "C:\\Users\\sunmoon\\IdeaProjects\\IOT_Back_Server\\model\\EMG_model.pkl"
EMG_SCALER_PATH = "C:\\Users\\sunmoon\\IdeaProjects\\IOT_Back_Server\\model\\EMG_scaler.pkl"
EOG_MODEL_PATH = "C:\\Users\\sunmoon\\IdeaProjects\\IOT_Back_Server\\model\\EOG_model.pkl"
EOG_SCALER_PATH = "C:\\Users\\sunmoon\\IdeaProjects\\IOT_Back_Server\\model\\EOG_scaler.pkl"
GSR_MODEL_PATH = "C:\\Users\\sunmoon\\IdeaProjects\\IOT_Back_Server\\model\\GSR_model.pkl"
GSR_SCALER_PATH = "C:\\Users\\sunmoon\\IdeaProjects\\IOT_Back_Server\\model\\GSR_scaler.pkl"
AirFlow_MODEL_PATH = "C:\\Users\\sunmoon\\IdeaProjects\\IOT_Back_Server\\model\\Airflow_model.pkl"
AirFlow_SCALER_PATH = "C:\\Users\\sunmoon\\IdeaProjects\\IOT_Back_Server\\model\\Airflow_scaler.pkl"


ECG_model = joblib.load(ECG_MODEL_PATH)  # Isolation Forest 모델 로드
ECG_scaler = joblib.load(ECG_SCALER_PATH)  # Scaler 로드
EMG_model = joblib.load(EMG_MODEL_PATH)
EMG_scaler = joblib.load(EMG_SCALER_PATH)
EOG_model = joblib.load(EOG_MODEL_PATH)
EOG_scaler = joblib.load(EOG_SCALER_PATH)
GSR_model = joblib.load(GSR_MODEL_PATH)
GSR_scaler = joblib.load(GSR_SCALER_PATH)
AirFlow_model = joblib.load(AirFlow_MODEL_PATH)
AirFlow_scaler = joblib.load(AirFlow_SCALER_PATH)

# 입력 데이터 형식 정의
class ECG_InputData(BaseModel):
    userid: str  # 사용자 ID
    ecgdata: list[float]  # 실수 데이터 리스트 (15,000개 예)

class EMG_InputData(BaseModel):
    userid: str  # 사용자 ID
    emgdata: list[float]  # 실수 데이터 리스트 (15,000개 예)

class EOG_InputData(BaseModel):
    userid: str  # 사용자 ID
    eogdata: list[float]  # 실수 데이터 리스트 (15,000개 예)

class GSR_InputData(BaseModel):
    userid: str  # 사용자 ID
    gsrdata: list[float]  # 실수 데이터 리스트 (15,000개 예)

class AirFlow_InputData(BaseModel):
    userid: str  # 사용자 ID
    airflowdata: list[float]  # 실수 데이터 리스트 (15,000개 예)

@app.post("/ecg")
async def predict(data: ECG_InputData):
    # 데이터를 numpy 배열로 변환
    input_features = np.array([data.ecgdata])  # 2D 배열로 변환 (1행 n열)
    
    # 스케일링
    scaled_data = ECG_scaler.transform(input_features)
    
    # 예측
    prediction = ECG_model.predict(scaled_data)
    result = "정상" if prediction[0] == 1 else "비정상"
    
    # 결과 반환
    return {
        "userid": data.userid,
        "ecgresult": result
    }

@app.post("/emg")
async def predict(data: EMG_InputData):
    # 데이터를 numpy 배열로 변환
    input_features = np.array([data.emgdata])  # 2D 배열로 변환 (1행 n열)
    
    # 스케일링
    scaled_data = EMG_scaler.transform(input_features)
    
    # 예측
    prediction = EMG_model.predict(scaled_data)
    result = "정상" if prediction[0] == 1 else "비정상"
    
    # 결과 반환
    return {
        "userid": data.userid,
        "emgresult": result
    }

@app.post("/eog")
async def predict(data: EOG_InputData):
    # 데이터를 numpy 배열로 변환
    input_features = np.array([data.eogdata])  # 2D 배열로 변환 (1행 n열)
    
    # 스케일링
    scaled_data = EOG_scaler.transform(input_features)
    
    # 예측
    prediction = EOG_model.predict(scaled_data)
    result = "정상" if prediction[0] == 1 else "비정상"
    
    # 결과 반환
    return {
        "userid": data.userid,
        "eogresult": result
    }

@app.post("/gsr")
async def predict(data: GSR_InputData):
    # 데이터를 numpy 배열로 변환
    input_features = np.array([data.gsrdata])  # 2D 배열로 변환 (1행 n열)
    
    # 스케일링
    scaled_data = GSR_scaler.transform(input_features)
    
    # 예측
    prediction = GSR_model.predict(scaled_data)
    result = "정상" if prediction[0] == 1 else "비정상"
    
    # 결과 반환
    return {
        "userid": data.userid,
        "gsrresult": result
    }

@app.post("/airflow")
async def predict(data: AirFlow_InputData):
    # 데이터를 numpy 배열로 변환
    input_features = np.array([data.airflowdata])  # 2D 배열로 변환 (1행 n열)
    
    # 스케일링
    scaled_data = AirFlow_scaler.transform(input_features)
    
    # 예측
    prediction = AirFlow_model.predict(scaled_data)
    result = "정상" if prediction[0] == 1 else "비정상"
    
    # 결과 반환
    return {
        "userid": data.userid,
        "airflowresult": result
    }