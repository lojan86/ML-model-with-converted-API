import uvicorn
from fastapi import FastAPI
from Banknote import Banknote
import pickle

app = FastAPI()

with open('classifier.pkl', 'rb') as pickle_in:
    classifier = pickle.load(pickle_in)
    
@app.post('/predict')
def predict_banknote(banknote: Banknote):
    data = banknote.dict()
    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy = data['entropy']
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    if prediction[0] == 1:
        result = 'Fake note'
    else:
        result = 'Authentic note'
    return {'prediction': result}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)