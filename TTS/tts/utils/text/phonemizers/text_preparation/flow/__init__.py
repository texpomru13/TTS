from TTS.tts.utils.text.phonemizers.text_preparation.flow.executor import FlowExecutor
from TTS.tts.utils.text.phonemizers.text_preparation.flow.mutators import GeneralTextPreprocessing

from TTS.tts.utils.text.phonemizers.text_preparation.flow.mutators.ru import URLVocalizer as URLVocalizerRu
from TTS.tts.utils.text.phonemizers.text_preparation.flow.mutators.ru import PhoneVocalizer as PhoneVocalizerRu

executor = FlowExecutor(
    GeneralTextPreprocessing,
    URLVocalizerRu,
    PhoneVocalizerRu
)

__all__ = [executor]
