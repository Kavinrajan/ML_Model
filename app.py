import uvicorn
from fastapi import FastAPI
from Accountnote import AccountNote
import numpy as np
import pickle
import pandas as pd

app = FastAPI()
pickle_in = open("classify.pkl", "rb")
classify = pickle.load(pickle_in)

@app.post("/predict")
def predict_account_note(account_note: AccountNote):
    data = account_note.dict()
    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy = data['entropy']
    # print(classifier.predict([[variance, skewness, curtosis, entropy]]))
    input_data = np.array([[variance, skewness, curtosis, entropy]])
    prediction = classify.predict(input_data)
    if(prediction[0]>0.5):
        prediction="Fake account note"
    else:
        prediction="Genuine account note"
    return {
        'prediction': prediction
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5001) 