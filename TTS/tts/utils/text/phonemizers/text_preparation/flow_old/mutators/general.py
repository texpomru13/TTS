from urllib.parse import unquote

from TTS.tts.utils.text.phonemizers.text_preparation.flow.mutators.base import AbstractMutator


class GeneralTextPreprocessing(AbstractMutator):
    handlers = [
        unquote
    ]

    def __call__(self, text: str) -> str:
        for func in self.handlers:
            text = func(text)

        return text
