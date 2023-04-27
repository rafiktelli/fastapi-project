from transformers import AutoTokenizer, CamembertForSequenceClassification, AutoModelForSequenceClassification
#from flask import Flask, request, jsonify
import torch
from fastapi import FastAPI, Request
import uvicorn
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# maybe you should add __name__

tokenizer = AutoTokenizer.from_pretrained("pekoDama/enlilia-french-text-classifier")
model = AutoModelForSequenceClassification.from_pretrained("pekoDama/enlilia-french-text-classifier")

@app.get('/home')
def get_hello():
    return "Hello world!"

@app.post('/predict')
async def predict(request: Request):
    data = await request.json()
    text = data['text']
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    # predicted_label = torch.argmax(outputs.logits).item()
    probabilities = torch.softmax(logits, dim=1)

    # Convert probabilities to percentages
    percentages = probabilities * 100
    return {'percentages': percentages.tolist()}






if __name__ == '__main__':
    uvicorn.run("deployfastapi:app")