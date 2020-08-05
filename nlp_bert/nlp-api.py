from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from nlp_bert.classifier import HateClassifier

app = FastAPI()
cls_model = HateClassifier()

Vector = List[float]


class Request(BaseModel):
    text: List[str]


class Response(BaseModel):
    predictions: List[Vector]
    embeddings: Optional[List[Vector]] = None


@app.get("/")
def root():
    return {"message": "API running"}


@app.post("/model/predict")
def predict(body: Request, embeddings: bool = False):
    # will process 5 sentences max
    text = body.text[:5]

    if isinstance(cls_model.model, Exception):
        raise cls_model.model

    probas, embeds = cls_model.process(text)

    return Response(
        predictions=probas,
        embeddings=embeds if embeddings else None
    )
