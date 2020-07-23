Summarization is the NLP task of compressing one or many documents but still retain the input's original context and 
meaning.

Below, we'll walk through how we can use AdaptNLP's `EasySummarizer` module to summarize large amounts of text with
state-of-the-art models.


## Getting Started with `EasySummarizer`

We'll first get started by importing the `EasySummarizer` class from AdaptNLP.

After that, we set some example text that we'll use further down, and then instantiate the summarizer.

```python
from adaptnlp import EasySummarizer

# Text from encyclopedia Britannica on Einstein
text = ["""Einstein’s education was disrupted by his father’s repeated failures at business. In 1894, after his company failed to get an important 
          contract to electrify the city of Munich, Hermann Einstein moved to Milan to work with a relative. Einstein was left at a boardinghouse in 
          Munich and expected to finish his education. Alone, miserable, and repelled by the looming prospect of military duty when he turned 16, Einstein 
          ran away six months later and landed on the doorstep of his surprised parents. His parents realized the enormous problems that he faced as a 
          school dropout and draft dodger with no employable skills. His prospects did not look promising.
          Fortunately, Einstein could apply directly to the Eidgenössische Polytechnische Schule (“Swiss Federal Polytechnic School”; in 1911, 
          following expansion in 1909 to full university status, it was renamed the Eidgenössische Technische Hochschule, or “Swiss Federal 
          Institute of Technology”) in Zürich without the equivalent of a high school diploma if he passed its stiff entrance examinations. His marks 
          showed that he excelled in mathematics and physics, but he failed at French, chemistry, and biology. Because of his exceptional math scores, 
          he was allowed into the polytechnic on the condition that he first finish his formal schooling. He went to a special high school run by 
          Jost Winteler in Aarau, Switzerland, and graduated in 1896. He also renounced his German citizenship at that time. (He was stateless until 1901, 
          when he was granted Swiss citizenship.) He became lifelong friends with the Winteler family, with whom he had been boarding. (Winteler’s 
          daughter, Marie, was Einstein’s first love; Einstein’s sister, Maja, would eventually marry Winteler’s son Paul; and his close friend Michele 
          Besso would marry their eldest daughter, Anna.)""",
       """Einstein would write that two “wonders” deeply affected his early years. The first was his encounter with a compass at age five. 
          He was mystified that invisible forces could deflect the needle. This would lead to a lifelong fascination with invisible forces. 
          The second wonder came at age 12 when he discovered a book of geometry, which he devoured, calling it his 'sacred little geometry 
          book'. Einstein became deeply religious at age 12, even composing several songs in praise of God and chanting religious songs on 
          the way to school. This began to change, however, after he read science books that contradicted his religious beliefs. This challenge 
          to established authority left a deep and lasting impression. At the Luitpold Gymnasium, Einstein often felt out of place and victimized 
          by a Prussian-style educational system that seemed to stifle originality and creativity. One teacher even told him that he would 
          never amount to anything."""]

summarizer = EasySummarizer()
```

### Summarizing with `summarize(text: str, model_name_or_path: str, mini_batch_size: int, num_beams:int, min_length: 
int, max_length: int, early_stopping: bool **kwargs)`

Now that we have the summarizer instantiated, we are ready to load in a model and compress the text 
with the built-in `summarize()` method.  

This method takes in parameters: `text`, `model_name_or_path`, and `mini_batch_size` as well as optional keyword arguments
from the `Transformers.PreTrainedModel.generate()` method.

!!! note 
    You can set `model_name_or_path` to any of Transformers pretrained Summarization Models with Language Model heads.
    Transformers models are located at [https://huggingface.co/models](https://huggingface.co/models).  You can also pass in
    the path of a custom trained Transformers `xxxWithLMHead` model.
 
The method returns a list of Strings.

Here is one example using the T5-small model:

```python
# Summarize
summaries = summarizer.summarize(text = text, model_name_or_path="t5-small", mini_batch_size=1, num_beams = 4, min_length=0, max_length=100, early_stopping=True)

print("Summaries:\n")
for s in summaries:
    print(s, "\n")
```

Another example is shown below using the Bart-large trained on CNN data:

```python
# Summarize
summaries = summarizer.summarize(text = text, model_name_or_path="bart-large-cnn", mini_batch_size=1, num_beams = 2, min_length=40, max_length=300, early_stopping=True)

print("Summaries:\n")
for s in summaries:
    print(s, "\n")
```

Below are some examples of Hugging Face's Pre-Trained Summarization models that you can use (These do
not include models hosted in Hugging Face's model repo):

| Model |  ID  |
| ----- | ---- |
| T5    |   't5-small'   |
|       |   't5-base'    |
|       |   't5-large'   | 
|       |   't5-3B'      |
|       |   't5-11B'     |
| Bart  |   'bart-large-cnn' |
