from adaptnlp import EasySummarizer


def test_easy_summarizer():
    summarizer = EasySummarizer()
    summarizer.summarize(text="Testing summarizer")
