from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import PreTrainedModel
from torch.nn import Softmax
import torch

from fastapi import HTTPException

from typing import List, Union
import zipfile
import glob
import wget
import os

PRE_TRAINED_MODEL = "distilbert-base-uncased"
MODEL_DIR = f"{os.getcwd()}/bertEp2"
MODEL_S3_URL = "https://" \
               "cc6e750869d1bf4c575d93c62ceaffbd880f62fdc70d92005eedad24f5865" \
               ".s3.amazonaws.com/bertEp2.zip"

ResponsePair = (List[List[float]], List[List[float]])


class HateClassifier:
    def __init__(self, model_dir=MODEL_DIR, pre_trained=PRE_TRAINED_MODEL):
        self.model: Union[Exception, PreTrainedModel] = self._init_model(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(pre_trained)
        self.probaLayer = Softmax(-1)

    def process(self, text: List[str]) -> ResponsePair:
        """Return binary predictions and embeddings from CLS token"""
        tokenized = self.tokenizer(text, padding=True, return_tensors='pt', truncation=True)
        out = self.model(**tokenized, output_hidden_states=True)

        probas, embeds = Softmax(-1)(out[0]), out[1][-1][:, 0]

        return probas.tolist(), embeds.tolist()

    def process_f(self, text: List[str], emb_size=768) -> ResponsePair:
        """Generate random probabilities and embeddings. Used for testing."""
        s = len(text)
        probas = torch.rand([s, 2]).tolist()
        embeddings = torch.rand([s, emb_size]).tolist()

        return probas, embeddings

    def _init_model(self, model_dir: str):
        self._download_model()

        try:
            return AutoModelForSequenceClassification.from_pretrained(model_dir)
        except:
            return HTTPException(status_code=501, detail="Unable to load model.")

    def _download_model(self, model_dir=MODEL_DIR, url=MODEL_S3_URL):
        if any(glob.glob(model_dir + "/*.bin")):
            return

        not os.path.isdir(model_dir) and os.mkdir(model_dir)
        model_path = wget.download(url=url, out=model_dir)

        with zipfile.ZipFile(model_path, 'r') as zip_ref:
            zip_ref.extractall(model_dir)

        os.remove(model_path)
