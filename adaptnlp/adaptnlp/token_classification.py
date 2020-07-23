import logging
from typing import List, Dict, Union
from collections import defaultdict

import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader

from flair.data import Sentence
from flair.models import SequenceTagger
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    PreTrainedTokenizer,
    PreTrainedModel,
    BertForSequenceClassification,
    XLNetForSequenceClassification,
    AlbertForSequenceClassification,
)

from tqdm import tqdm

from adaptnlp.model import AdaptiveModel


logger = logging.getLogger(__name__)


class TransformersTokenTagger(AdaptiveModel):
    """ Adaptive model for Transformer's Token Tagger Model

    Usage:
    ```python
    >>> tagger = TransformersTokenTagger.load("dbmdz/bert-large-cased-finetuned-conll03-english")
    >>> tagger.predict(text="Example text", mini_batch_size=32)
    ```

    **Parameters:**

    * **tokenizer** - A tokenizer object from Huggingface's transformers (TODO)and tokenizers
    * **model** - A transformers token tagger model
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
        """ Class method for loading and constructing this tagger 

        * **model_name_or_path** - A key string of one of Transformer's pre-trained Token Tagger Model
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)
        tagger = cls(tokenizer, model)
        return tagger

    def predict(
        self,
        text: Union[List[str], str],
        mini_batch_size: int = 32,
        grouped_entities: bool = True,
        **kwargs,
    ) -> List[List[Dict]]:
        """ Predict method for running inference using the pre-trained token tagger model.
        Returns a list of lists of tagged entities.

        * **text** - String, list of strings, sentences, or list of sentences to run inference on
        * **mini_batch_size** - Mini batch size
        * **grouped_entities** - Set True to get whole entity span strings (Default True)
        * **&ast;&ast;kwargs**(Optional) - Optional arguments for the Transformers tagger 
        """
        if isinstance(text, str):
            text = [text]
        results: List[Dict] = []

        with torch.no_grad():
            # Tokenize and get dataset
            dataset = self._tokenize(text)
            dataloader = DataLoader(dataset, batch_size=mini_batch_size)

            logger.info(f"Running prediction on {len(dataset)} text sequences")
            logger.info(f"Batch size = {mini_batch_size}")
            for batch in tqdm(dataloader, desc="Predicting text"):
                self.model.eval()
                batch = tuple(t.to(self.device) for t in batch)

                if len(batch) == 3:
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "token_type_ids": batch[2],
                    }
                else:
                    inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
                outputs = self.model(**inputs)

                # Iterate through batch for tagged token predictions
                for idx, pred in enumerate(outputs[0]):
                    entities = pred.cpu().detach().numpy()
                    input_ids = inputs["input_ids"].cpu().numpy()[idx]
                    tagged_entities = self._generate_tagged_entities(
                        entities=entities,
                        input_ids=input_ids,
                        grouped_entities=grouped_entities,
                    )
                    results += tagged_entities

        return results

    def _tokenize(
        self, sentences: Union[List[Sentence], Sentence, List[str], str]
    ) -> TensorDataset:
        """ Batch tokenizes text and produces a `TensorDataset` with them """

        tokenized_text = self.tokenizer.batch_encode_plus(
            sentences, return_tensors="pt", pad_to_max_length=True,
        )

        # Bart, XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use token_type_ids
        if isinstance(
            self.model,
            (
                BertForSequenceClassification,
                XLNetForSequenceClassification,
                AlbertForSequenceClassification,
            ),
        ):
            dataset = TensorDataset(
                tokenized_text["input_ids"],
                tokenized_text["attention_mask"],
                tokenized_text["token_type_ids"],
            )
        else:
            dataset = TensorDataset(
                tokenized_text["input_ids"], tokenized_text["attention_mask"]
            )

        return dataset

    # `_group_entites` and `_generate_tagged_entities` modified from pipeline code snippet from Transformers
    def _group_entities(
        self, entities: List[dict], idx_start: int, idx_end: int
    ) -> Dict:
        """Returns grouped entities"""
        # Get the last entity in the entity group
        entity = entities[-1]["entity"]
        scores = np.mean([entity["score"] for entity in entities])
        tokens = [entity["word"] for entity in entities]

        entity_group = {
            "entity_group": entity,
            "score": np.mean(scores),
            "word": self.tokenizer.convert_tokens_to_string(tokens),
            "offsets": (idx_start, idx_end),
        }
        return entity_group

    def _generate_tagged_entities(
        self, entities: np.ndarray, input_ids: np.ndarray, grouped_entities: bool = True
    ) -> List[Dict]:
        """Generate full list of entities given tagged token predictions and input_ids"""

        score = np.exp(entities) / np.exp(entities).sum(-1, keepdims=True)
        labels_idx = score.argmax(axis=-1)

        answers = []
        entities = []
        entity_groups = []
        entity_group_disagg = []
        # Filter to labels not in `self.ignore_labels`
        filtered_labels_idx = [
            (idx, label_idx)
            for idx, label_idx in enumerate(labels_idx)
            if self.model.config.id2label[label_idx] not in ["O"]
        ]

        for idx, label_idx in filtered_labels_idx:
            # print(tokenizer.convert_ids_to_tokens(int(input_ids[idx])))
            entity = {
                "word": self.tokenizer.convert_ids_to_tokens(int(input_ids[idx])),
                "score": score[idx][label_idx].item(),
                "entity": self.model.config.id2label[label_idx],
                "index": idx,
            }
            last_idx, _ = filtered_labels_idx[-1]
            if grouped_entities:
                if not entity_group_disagg:
                    entity_group_disagg += [entity]
                    if idx == last_idx:
                        entity_groups += [
                            self._group_entities(
                                entity_group_disagg, idx - len(entity_group_disagg), idx
                            )
                        ]
                    continue

                # If the current entity is similar and adjacent to the previous entity, append it to the disaggregated entity group
                if (
                    entity["entity"] == entity_group_disagg[-1]["entity"]
                    and entity["index"] == entity_group_disagg[-1]["index"] + 1
                ):
                    entity_group_disagg += [entity]
                    # Group the entities at the last entity
                    if idx == last_idx:
                        entity_groups += [
                            self._group_entities(
                                entity_group_disagg, idx - len(entity_group_disagg), idx
                            )
                        ]
                # If the current entity is different from the previous entity, aggregate the disaggregated entity group
                else:
                    entity_groups += [
                        self._group_entities(
                            entity_group_disagg,
                            entity_group_disagg[-1]["index"] - len(entity_group_disagg),
                            entity_group_disagg[-1]["index"],
                        )
                    ]
                    entity_group_disagg = [entity]

            entities += [entity]

        # Append
        if grouped_entities:
            answers += [entity_groups]
        else:
            answers += [entities]

        return answers


class FlairTokenTagger(AdaptiveModel):
    """ Adaptive Model for Flair's Token Tagger...very basic

    Usage:
    ```python
    >>> tagger = FlairTokenTagger.load("en-sentiment")
    >>> tagger.predict(text="Example text", mini_batch_size=32)
    ```

    **Parameters:**

    * **model_name_or_path** - A key string of one of Flair's pre-trained Token tagger Model
    """

    def __init__(self, model_name_or_path: str):
        self.tagger = SequenceTagger.load(model_name_or_path)

    @classmethod
    def load(cls, model_name_or_path: str) -> AdaptiveModel:
        """ Class method for loading a constructing this tagger 

        * **model_name_or_path** - A key string of one of Flair's pre-trained Token tagger Model
        """
        tagger = cls(model_name_or_path)
        return tagger

    def predict(
        self,
        text: Union[List[Sentence], Sentence, List[str], str],
        mini_batch_size: int = 32,
        use_tokenizer=True,
        **kwargs,
    ) -> List[Sentence]:
        """ Predict method for running inference using the pre-trained token tagger model

        * **text** - String, list of strings, sentences, or list of sentences to run inference on
        * **mini_batch_size** - Mini batch size
        * **&ast;&ast;kwargs**(Optional) - Optional arguments for the Flair tagger 
        """
        return self.tagger.predict(
            sentences=text,
            mini_batch_size=mini_batch_size,
            use_tokenizer=use_tokenizer,
            **kwargs,
        )


class EasyTokenTagger:
    """ Token level classification models

    Usage:

    ```python
    >>> tagger = adaptnlp.EasyTokenTagger()
    >>> tagger.tag_text(text="text you want to tag", model_name_or_path="ner-ontonotes")
    ```
    """

    def __init__(self):
        self.token_taggers: Dict[AdaptiveModel] = defaultdict(bool)

    def tag_text(
        self,
        text: Union[List[Sentence], Sentence, List[str], str],
        model_name_or_path: str = "ner-ontonotes",
        mini_batch_size: int = 32,
        **kwargs,
    ) -> List[Sentence]:
        """ Tags tokens with labels the token classification models have been trained on

        * **text** - Text input, it can be a string or any of Flair's `Sentence` input formats
        * **model_name_or_path** - The hosted model name key or model path
        * **mini_batch_size** - The mini batch size for running inference
        * **&ast;&ast;kwargs** - Keyword arguments for Flair's `SequenceTagger.predict()` method
        **return** - A list of Flair's `Sentence`'s
        """
        # Load Sequence Tagger Model and Pytorch Module into tagger dict
        if not self.token_taggers[model_name_or_path]:
            """
            self.token_taggers[model_name_or_path] = SequenceTagger.load(
                model_name_or_path
            )
            """
            # TODO: Find an alternative model-check method like an `is_available(model_name_or_path)
            # Check whether this is a Transformers or Flair Sequence Classifier model we're loading
            try:
                self.token_taggers[model_name_or_path] = FlairTokenTagger.load(
                    model_name_or_path
                )
            except FileNotFoundError:
                logger.info(
                    f"{model_name_or_path} not a valid Flair pre-trained model...checking transformers repo"
                )
                try:
                    self.token_taggers[
                        model_name_or_path
                    ] = TransformersTokenTagger.load(model_name_or_path)
                except ValueError:
                    logger.info("Not a valid model_name_or_path param")
                    return [Sentence("")]

        tagger = self.token_taggers[model_name_or_path]
        return tagger.predict(
            text=text, mini_batch_size=mini_batch_size, use_tokenizer=True, **kwargs,
        )

    def tag_all(
        self,
        text: Union[List[Sentence], Sentence, List[str], str],
        mini_batch_size: int = 32,
        **kwargs,
    ) -> List[Sentence]:
        """ Tags tokens with all labels from all token classification models

        * **text** - Text input, it can be a string or any of Flair's `Sentence` input formats
        * **mini_batch_size** - The mini batch size for running inference
        * **&ast;&ast;kwargs** - Keyword arguments for Flair's `SequenceTagger.predict()` method
        **return** A list of Flair's `Sentence`'s
        """
        if len(self.token_taggers) == 0:
            print("No token classification models loaded...")
            return Sentence()
        sentences = text
        for tagger_name in self.token_taggers.keys():
            sentences = self.tag_text(
                sentences,
                model_name_or_path=tagger_name,
                mini_batch_size=mini_batch_size,
                **kwargs,
            )
        return sentences
