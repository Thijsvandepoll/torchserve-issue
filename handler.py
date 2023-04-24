import logging
from abc import ABC

import torch
from tokenizers import Regex, normalizers
from tokenizers.normalizers import NFKD, Lowercase, Replace, Strip, StripAccents
from transformers import AutoModel, AutoTokenizer, __version__
from ts.torch_handler.text_handler import BaseHandler

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", __version__)


def init_normalizer():
    return normalizers.Sequence(
        [
            Lowercase(),
            NFKD(),
            StripAccents(),
            Strip(),
        ]
    )


class TransformersHandler(BaseHandler, ABC):
    tokenizer_kwargs = {"truncation": True, "padding": True}

    def __init__(self):
        super().__init__()
        self.initialized = False
        self.tokenizer = None
        self.normalizer = init_normalizer()

    def initialize(self, context):
        properties = context.system_properties
        self.map_location = (
            "cuda"
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        self.device = torch.device(
            self.map_location + ":" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else self.map_location
        )
        self.manifest = context.manifest

        model_dir = properties.get("model_dir")
        self.model = AutoModel.from_pretrained(model_dir)
        if self.model is not None:
            logger.info("Successfully loaded model")
        else:
            raise RuntimeError("Missing model")

        self.model.to(self.device)
        self.model.eval()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if self.tokenizer is not None:
            logger.info("Successfully loaded tokenizer")
        else:
            raise RuntimeError("Missing tokenizer")

        self.initialized = True

    def preprocess(self, data):
        if all(isinstance(k, dict) for k in data):
            # Assume when passing instances, it is length 1
            data = [k.get("body")["instances"][0] for k in data]

        # Normalize
        normalized = [self.normalizer.normalize_str(k) for k in data]

        # Tokenize
        tokenized = self.tokenizer(
            normalized, return_tensors="pt", **self.tokenizer_kwargs
        )
        return tokenized

    def inference(self, data, *args, **kwargs):
        with torch.no_grad():
            marshalled_data = data.to(self.device)
            results = self.model(**marshalled_data, **kwargs)
        return results

    def postprocess(self, data):
        # Compute embedding for [CLS] token
        result = data["last_hidden_state"][:, 0]
        return result.tolist()
