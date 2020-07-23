import logging
from typing import List, Dict, Union
from collections import defaultdict

import torch
from torch.utils.data import TensorDataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelWithLMHead,
    PreTrainedTokenizer,
    PreTrainedModel,
    T5ForConditionalGeneration,
    BartForConditionalGeneration,
)

from tqdm import tqdm

from adaptnlp.model import AdaptiveModel

logger = logging.getLogger(__name__)


class TransformersSummarizer(AdaptiveModel):
    """ Adaptive model for Transformer's Conditional Generation or Language Models (Transformer's T5 and Bart
        conditiional generation models have a language modeling head)

        Usage:
        ```python
        >>> summarizer = TransformersSummarizer.load("transformers-summarizer-model")
        >>> summarizer.predict(text="Example text", mini_batch_size=32)
        ```

        **Parameters:**

        * **tokenizer** - A tokenizer object from Huggingface's transformers (TODO)and tokenizers
        * **model** - A transformers Conditional Generation (Bart or T5) or Language model
        """

    def __init__(self, tokenizer: PreTrainedTokenizer, model: PreTrainedModel):
        # Load up model and tokenizer
        self.tokenizer = tokenizer
        self.model = model

        # Setup cuda and automatic allocation of model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

    @classmethod
    def load(cls, model_name_or_path: str) -> AdaptiveModel:
        """ Class method for loading and constructing this classifier

         * **model_name_or_path** - A key string of one of Transformer's pre-trained Summarizer Model
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelWithLMHead.from_pretrained(model_name_or_path)
        summarizer = cls(tokenizer, model)
        return summarizer

    def predict(
        self,
        text: Union[List[str], str],
        mini_batch_size: int = 32,
        num_beams: int = 4,
        min_length: int = 0,
        max_length: int = 128,
        early_stopping: bool = True,
        **kwargs,
    ) -> List[str]:
        """ Predict method for running inference using the pre-trained sequence classifier model

        * **text** - String, list of strings, sentences, or list of sentences to run inference on
        * **mini_batch_size** - Mini batch size
        * **num_beams** - Number of beams for beam search. Must be between 1 and infinity. 1 means no beam search.  Default to 4.
        * **min_length** -  The min length of the sequence to be generated. Default to 0
        * **max_length** - The max length of the sequence to be generated. Between min_length and infinity. Default to 128
        * **early_stopping** - if set to True beam search is stopped when at least num_beams sentences finished per batch.
        * **&ast;&ast;kwargs**(Optional) - Optional arguments for the Transformers `PreTrainedModel.generate()` method
        """
        with torch.no_grad():

            # Make all inputs lists
            if isinstance(text, str):
                text = [text]

            # T5 requires "summarize: " precursor text for pre-trained summarizer
            if isinstance(self.model, T5ForConditionalGeneration):
                text = [f"summarize: {t}" for t in text]

            dataset = self._tokenize(text)
            dataloader = DataLoader(dataset, batch_size=mini_batch_size)
            summaries = []

            logger.info(f"Running summarizer on {len(dataset)} text sequences")
            logger.info(f"Batch size = {mini_batch_size}")
            for batch in tqdm(dataloader, desc="Summarizing"):
                self.model.eval()
                batch = tuple(t.to(self.device) for t in batch)

                if len(batch) == 3:
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "token_type_ids": batch[2],
                    }
                else:
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                    }
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    num_beams=num_beams,
                    min_length=min_length,
                    max_length=max_length,
                    early_stopping=early_stopping,
                    **kwargs,
                )

                for o in outputs:
                    summaries.append(
                        [
                            self.tokenizer.decode(
                                o,
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=False,
                            )
                        ].pop()
                    )

        return summaries

    def _tokenize(self, text: Union[List[str], str]) -> TensorDataset:
        """ Batch tokenizes text and produces a `TensorDataset` with text """

        # Pre-trained Bart summarization model has a max length fo 1024 tokens for input
        if isinstance(self.model, BartForConditionalGeneration):
            tokenized_text = self.tokenizer.batch_encode_plus(
                text,
                return_tensors="pt",
                max_length=1024,
                add_special_tokens=True,
                pad_to_max_length=True,
            )
        else:
            tokenized_text = self.tokenizer.batch_encode_plus(
                text,
                return_tensors="pt",
                pad_to_max_length=True,
                add_special_tokens=True,
            )

        # Bart doesn't use `token_type_ids`
        if isinstance(self.model, T5ForConditionalGeneration):
            dataset = TensorDataset(
                tokenized_text["input_ids"],
                tokenized_text["attention_mask"],
                tokenized_text["token_type_ids"],
            )
        else:
            dataset = TensorDataset(
                tokenized_text["input_ids"], tokenized_text["attention_mask"],
            )

        return dataset


class EasySummarizer:
    """ Summarization Module

    Usage:

    ```python
    >>> summarizer = EasySummarizer()
    >>> summarizer.summarize(text="Summarize this text", model_name_or_path="t5-small")
    ```

    """

    def __init__(self):
        self.summarizers: Dict[AdaptiveModel] = defaultdict(bool)

    def summarize(
        self,
        text: Union[List[str], str],
        model_name_or_path: str = "t5-small",
        mini_batch_size: int = 32,
        num_beams: int = 4,
        min_length: int = 0,
        max_length: int = 128,
        early_stopping: bool = True,
        **kwargs,
    ) -> List[str]:
        """ Predict method for running inference using the pre-trained sequence classifier model

        * **text** - String, list of strings, sentences, or list of sentences to run inference on
        * **model_name_or_path** - A String model id or path to a pre-trained model repository or custom trained model directory 
        * **mini_batch_size** - Mini batch size
        * **num_beams** - Number of beams for beam search. Must be between 1 and infinity. 1 means no beam search.  Default to 4.
        * **min_length** -  The min length of the sequence to be generated. Default to 0
        * **max_length** - The max length of the sequence to be generated. Between min_length and infinity. Default to 128
        * **early_stopping** - if set to True beam search is stopped when at least num_beams sentences finished per batch.
        * **&ast;&ast;kwargs**(Optional) - Optional arguments for the Transformers `PreTrainedModel.generate()` method
        """
        if not self.summarizers[model_name_or_path]:
            self.summarizers[model_name_or_path] = TransformersSummarizer.load(
                model_name_or_path
            )

        summarizer = self.summarizers[model_name_or_path]
        return summarizer.predict(
            text=text,
            mini_batch_size=mini_batch_size,
            num_beams=num_beams,
            min_length=min_length,
            max_length=max_length,
            early_stopping=early_stopping,
            **kwargs,
        )
