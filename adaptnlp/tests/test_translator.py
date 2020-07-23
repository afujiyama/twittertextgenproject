from adaptnlp import EasyTranslator


def test_easy_Translator():
    translator = EasyTranslator()
    translator.translate(text="Testing summarizer")
