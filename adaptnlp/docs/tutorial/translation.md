Translation is the task of producing the input text in another language.

Below, we'll walk through how we can use AdaptNLP's `EasyTranslator` module to translate text with
state-of-the-art models.


## Getting Started with `EasyTranslator`

We'll first get started by importing the `EasyTranslator` class from AdaptNLP.

After that, we set some example text that we'll use further down, and then instantiate the translator.

```python
from adaptnlp import EasyTranslator

text = ["Machine learning will take over the world very soon.",
        "Machines can speak in many languages.",]

translator = EasyTranslator()
```

### Translating with `translate(text: str, model_name_or_path: str, mini_batch_size: int, num_beams:int, min_length: 
int, max_length: int, early_stopping: bool **kwargs)`

Now that we have the translator instantiated, we are ready to load in a model and translate the text 
with the built-in `translate()` method.  

This method takes in parameters: `text`, `model_name_or_path`, and `mini_batch_size` as well as optional keyword arguments
from the `Transformers.PreTrainedModel.generate()` method.

!!! note 
    You can set `model_name_or_path` to any of Transformers pretrained Translation Models with Language Model heads.
    Transformers models are located at [https://huggingface.co/models](https://huggingface.co/models).  You can also pass in
    the path of a custom trained Transformers `xxxWithLMHead` model.
 
The method returns a list of Strings.

Here is an example using the T5-small model:

```python
# Translate
translations = translator.translate(text = text, t5_prefix="translate English to German", model_name_or_path="t5-small", mini_batch_size=1, min_length=0, max_length=100, early_stopping=True)

print("Translations:\n")
for t in translations:
    print(t, "\n")
```

Below are some examples of Hugging Face's Pre-Trained Translation models that you can use (These do
not include models hosted in Hugging Face's model repo):

| Model |  ID  |
| ----- | ---- |
| T5    |   't5-small'   |
|       |   't5-base'    |
|       |   't5-large'   | 
|       |   't5-3B'      |
|       |   't5-11B'     |
