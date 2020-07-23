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
)

from tqdm import tqdm

from adaptnlp.model import AdaptiveModel

logger = logging.getLogger(__name__)


class TransformersTextGenerator(AdaptiveModel):
    """ Adaptive model for Transformer's Language Models 

        Usage:
        ```python
        >>> generator = TransformersTextGenerator.load("gpt2")
        >>> generator.generate(text="Example text", mini_batch_size=32)
        ```

        **Parameters:**

        * **tokenizer** - A tokenizer object from Huggingface's transformers (TODO)and tokenizers
        * **model** - A transformers Language model
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
        """ Class method for loading and constructing this Model 

         * **model_name_or_path** - A key string of one of Transformer's pre-trained Language Model
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, pad_token="<PAD>")
        model = AutoModelWithLMHead.from_pretrained(model_name_or_path)
        generator = cls(tokenizer, model)
        return generator

    def predict(
        self,
        text: Union[List[str], str],
        mini_batch_size: int = 32,
        num_tokens_to_produce: int = 50,
        **kwargs,
    ) -> List[str]:
        """ Predict method for running inference using the pre-trained sequence classifier model.  Keyword arguments
        for parameters of the method `Transformers.PreTrainedModel.generate()` can be used as well.

        * **text** - String, list of strings, sentences, or list of sentences to run inference on
        * **mini_batch_size** - Mini batch size
        * **num_tokens_to_produce** - Number of tokens you want to generate
        * **&ast;&ast;kwargs**(Optional) - Optional arguments for the Transformers `PreTrainedModel.generate()` method
        """
        with torch.no_grad():

            # Make all inputs lists
            if isinstance(text, str):
                text = [text]

            dataset = self._tokenize(text)
            dataloader = DataLoader(dataset, batch_size=mini_batch_size)
            results = []

            logger.info(f"Running text generator on {len(dataset)} text sequences")
            logger.info(f"Batch size = {mini_batch_size}")
            for batch in tqdm(dataloader, desc="Generating"):
                self.model.eval()
                batch = tuple(t.to(self.device) for t in batch)

                if len(batch) == 3:
                    inputs = {
                        "input_ids": batch[0],
                        "attention_masks": batch[1],
                        "token_type_ids": batch[2],
                    }
                else:
                    inputs = {
                        "input_ids": batch[0],
                        "attention_masks": batch[1],
                    }
                # model.generate() does not have batch inference implemented yet
                generated_text = self._batch_generate(
                    inputs=inputs,
                    seq_len=batch[0].shape[1],
                    num_tokens_to_produce=num_tokens_to_produce,
                )
                results += generated_text

        return results

    def _tokenize(self, text: Union[List[str], str]) -> TensorDataset:
        """ Batch tokenizes text and produces a `TensorDataset` with text """

        tokenized_text = self.tokenizer.batch_encode_plus(
            text,
            return_tensors="pt",
            max_length=512,
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

    def _batch_generate(
        self, inputs: Dict, seq_len: int, num_tokens_to_produce: int
    ) -> List[str]:
        """Generates text data with varying text sizes"""
        input_ids = inputs["input_ids"]
        attn_mask = inputs["attention_masks"]

        pad_token_id = self.tokenizer.pad_token_id
        eos_token_id = self.tokenizer.eos_token_id
        eos_not_in_sents = torch.ones(input_ids.shape[0]).long().to(self.device)

        # we need to get the token ids of the last non-padded value
        last_non_masked_idx = torch.sum(attn_mask, dim=1) - 1
        start_idx = inp_idx = (
            (last_non_masked_idx)
            .view(-1, 1)
            .repeat(1, self.tokenizer.vocab_size)
            .unsqueeze(1)
        )
        past = None

        # get correct position ids
        position_ids = torch.tensor(
            [list(range(seq_len)) for i in range(input_ids.shape[0])]
        ).to(self.device)
        for i, position_ids_slice in enumerate(position_ids):
            position_ids_slice[last_non_masked_idx[i] :] = position_ids_slice[
                last_non_masked_idx[i]
            ]

        for step in range(num_tokens_to_produce):
            outputs = self.model(
                input_ids, attention_mask=attn_mask, position_ids=position_ids
            )

            # in the first decoding step, we want to use the 'real' last position for each sentence
            if step == 0:
                next_token_logits = outputs[0].gather(1, start_idx).squeeze(1)
            else:
                next_token_logits = outputs[0][:, -1, :]

            next_tokens = torch.argmax(next_token_logits, dim=-1)

            # this updates which sentences have not seen an <EOS> token so far
            # if one <EOS> token was seen the sentence is finished
            eos_not_in_sents.mul_(next_tokens.ne(eos_token_id).long())

            # either append a padding token here if <EOS> has been seen or append next token
            tokens_to_add = next_tokens * (eos_not_in_sents) + pad_token_id * (
                1 - eos_not_in_sents
            )

            # Update input_ids, attn_mask and position_ids
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            attn_mask = torch.cat(
                [attn_mask, torch.ones((attn_mask.shape[0], 1)).long().to(self.device)],
                dim=1,
            )
            position_ids = torch.cat(
                [position_ids, (position_ids[:, -1] + 1).unsqueeze(-1)], dim=1
            )

        return [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in input_ids
        ]


class EasyTextGenerator:
    """ Text Generation Module

    Usage:

    ```python
    >>> generator = EasyGenerator()
    >>> generator.generate(text="generate from this text", num_tokens_to_produce=50)
    ```

    """

    def __init__(self):
        self.generators: Dict[AdaptiveModel] = defaultdict(bool)

    def generate(
        self,
        text: Union[List[str], str],
        model_name_or_path: str = "gpt2",
        mini_batch_size: int = 32,
        num_tokens_to_produce: int = 50,
        **kwargs,
    ) -> List[str]:
        """ Predict method for running inference using the pre-trained sequence classifier model. Keyword arguments
        for parameters of the method `Transformers.PreTrainedModel.generate()` can be used as well.

        * **text** - String, list of strings, sentences, or list of sentences to run inference on
        * **model_name_or_path** - A String model id or path to a pre-trained model repository or custom trained model directory 
        * **mini_batch_size** - Mini batch size
        * **num_tokens_to_produce** - Number of tokens you want to generate
        * **&ast;&ast;kwargs**(Optional) - Optional arguments for the Transformers `PreTrainedModel.generate()` method
        """
        if not self.generators[model_name_or_path]:
            self.generators[model_name_or_path] = TransformersTextGenerator.load(
                model_name_or_path
            )

        generator = self.generators[model_name_or_path]
        return generator.predict(
            text=text,
            mini_batch_size=mini_batch_size,
            num_tokens_to_produce=num_tokens_to_produce,
            **kwargs,
        )
