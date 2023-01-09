import joblib
import uvicorn
from fastapi import FastAPI
from fastapi import Query
from app.model.stressDetect import StressDetect

app=FastAPI()

model=joblib.load(open('D:/M2 MBD/deep learning/projet/app/model/stressDetectorModel.pkl','rb'))

@app.get("/")
async def index():
    return {"message": "Hello World"}

@app.post("/detect")
def detect_stress(data:StressDetect):
    data=data.dict()
    lex_liwc_Tone=data['lex_liwc_Tone']
    lex_liwc_posemo=data['lex_liwc_posemo']
    lex_liwc_negemo=data['lex_liwc_negemo']
    lex_liwc_anx=['lex_liwc_anx']
    lex_liwc_Dic=['lex_liwc_Dic']
    sentiment=['sentiment']
    prediction=model.predict([[lex_liwc_Tone, lex_liwc_posemo, lex_liwc_negemo,lex_liwc_anx,lex_liwc_Dic,sentiment]])
    if(prediction[0]>0.5):
        prediction="stress"
    else:
        prediction="no_stress"
    return {
        'detection:': prediction
    }


if __name__ == '__main__':
    uvicorn.run(app,host="127.0.0.1",port=8000)