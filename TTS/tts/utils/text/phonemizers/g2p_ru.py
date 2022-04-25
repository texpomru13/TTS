import re
import os
import sys

from pathlib import Path
# from russian_g2p_neuro.src.model import Russian_G2P

from TTS.tts.utils.text.phonemizers.text_preparation.normalizers import russian_normalizer, g2p_ru_wrapper
from TTS.tts.utils.text.phonemizers.text_preparation.normalizers import accentizer_from_morpher_ru_wrapper, \
    stress_rnn_ru
from gruut import sentences
# from russian_g2p.Grapheme2Phoneme import Grapheme2Phoneme
# "ː"
import importlib
from typing import List

from TTS.tts.utils.text.phonemizers.base import BasePhonemizer
from TTS.tts.utils.text.punctuation import Punctuation

MODULE_PATH = Path(__file__).parents[0]
# g2p = Grapheme2Phoneme() # Russian_G2P(MODULE_PATH / 'russian_g2p_neuro/model')
letter = "квс"
glas = "уеёыаоэяию"
ph_gals = "ɨoeaiu"


def replacer(s, newstring, index, nofail=False):
    # raise an error if index is outside of the string
    if not nofail and index not in range(len(s)):
        raise ValueError("index outside given string")

    # if not erroring, but the index is still not in the correct range..
    if index < 0:  # add it to the beginning
        return newstring + s
    if index > len(s):  # add it to the end
        return s + newstring

    # insert the new string between "slices" of the original
    return s[:index] + newstring + s[index + 1:]


def transcribe_word_list(text, space=False):
    result = []
    text = russian_normalizer(text, replacing_symbols=False, expand_difficult_abbreviations=True,
                              use_stress_dictionary=False, use_g2p_accentor=False,
                              use_g2p=False)
    punct = ",.!?:;—"
    for point in punct:
        text = text.replace(point, " " + point + " ")

    wordlist = text.split()
    wordilst_ac = stress_rnn_ru.put_stress(accentizer_from_morpher_ru_wrapper(text), stress_symbol='+',
                                           accuracy_threshold=0.0).split()
    for word, word_ac in zip(wordlist, wordilst_ac):
        if word in punct:
            result.append(word)
            continue
        prediction = ""
        for sent in sentences(word, lang="ru"):
            for wor in sent:
                if wor.phonemes:
                    prediction = wor.phonemes

        idx = word_ac.find("+")
        pred = []
        if idx != -1:
            idx = len(re.sub(f"[{glas}]", "", word[:idx]))
            cnt = 0
            for ph in prediction:
                if "ː" in ph:
                    ph = ph.replace("ː", "")
                if ph in ph_gals:
                    cnt += 1
                    if cnt == idx:
                        pred.append(ph + "ː")
                        continue
                pred.append(ph)
            if "ː" in "".join(pred):
                result.append(" ".join(pred))
            elif "ː" in "".join(prediction):
                result.append(" ".join(prediction))
            else:
                result.append(" ".join(prediction) + "ː")
            continue

        else:
            if "ː" in "".join(prediction):
                result.append(" ".join(prediction))
            else:
                result.append(" ".join(prediction) + "ː")

    if space:
        return result
    else:
        return ' '.join(result).split()


# Table for str.translate to fix gruut/TTS phoneme mismatch

class G2p_ru(BasePhonemizer):
    """Gruut wrapper for G2P

    Args:
        language (str):
            Valid language code for the used backend.

        punctuations (str):
            Characters to be treated as punctuation. Defaults to `Punctuation.default_puncs()`.

        keep_puncs (bool):
            If true, keep the punctuations after phonemization. Defaults to True.

        use_espeak_phonemes (bool):
            If true, use espeak lexicons instead of default Gruut lexicons. Defaults to False.

        keep_stress (bool):
            If true, keep the stress characters after phonemization. Defaults to False.

    Example:

        >>> from TTS.tts.utils.text.phonemizers.gruut_wrapper import Gruut
        >>> phonemizer = Gruut('en-us')
        >>> phonemizer.phonemize("Be a voice, not an! echo?", separator="|")
        'b|i| ə| v|ɔ|ɪ|s, n|ɑ|t| ə|n! ɛ|k|o|ʊ?'
    """

    def __init__(
            self,
            language="ru",
            punctuations=Punctuation.default_puncs(),
            keep_puncs=True,
            use_espeak_phonemes=False,
            keep_stress=False,
    ):
        super().__init__(language, punctuations=punctuations, keep_puncs=keep_puncs)
        self.use_espeak_phonemes = use_espeak_phonemes
        self.keep_stress = keep_stress

    @staticmethod
    def name():
        return "g2p_ru"

    def phonemize_g2p_ru(self, text: str, separator: str = "|", tie=False) -> str:  # pylint: disable=unused-argument
        """Convert input text to phonemes.

        Gruut phonemizes the given `str` by seperating each phoneme character with `separator`, even for characters
        that constitude a single sound.

        It doesn't affect 🐸TTS since it individually converts each character to token IDs.

        Examples::
            "hello how are you today?" -> `h|ɛ|l|o|ʊ| h|a|ʊ| ɑ|ɹ| j|u| t|ə|d|e|ɪ`

        Args:
            text (str):
                Text to be converted to phonemes.

            tie (bool, optional) : When True use a '͡' character between
                consecutive characters of a single phoneme. Else separate phoneme
                with '_'. This option requires espeak>=1.49. Default to False.
        """

        # ph_words = [separator.join(word_phonemes) for word_phonemes in ph_list]
        if self.keep_stress:
            ph_lists = transcribe_word_list(text,True)
            ph_words = [word.replace(" ", separator) for word in ph_lists]
            ph = f"{separator} ".join(ph_words)
        else:
            ph_words = transcribe_word_list(text)
            ph = f"{separator}".join(ph_words)
        return ph

    def _phonemize(self, text, separator):
        return self.phonemize_g2p_ru(text, separator, tie=False)

    def is_supported_language(self, language):
        """Returns True if `language` is supported by the backend"""
        return language == "ru" or language == "ru-ru"

    @staticmethod
    def supported_languages() -> List:
        """Get a dictionary of supported languages.

        Returns:
            List: List of language codes.
        """
        return ["ru"]

    def version(self):
        """Get the version of the used backend.

        Returns:
            str: Version of the used backend.
        """
        return "0"

    @classmethod
    def is_available(cls):
        """Return true if ESpeak is available else false"""
        return True


if __name__ == "__main__":
    e = G2p_ru(language="ru")
    print(e.supported_languages())
    print(e.version())
    print(e.language)
    print(e.name())
    print(e.is_available())

    e = G2p_ru(language="ru", keep_puncs=False)
    print("`" + e.phonemize("hello how are you today?") + "`")

    e = G2p_ru(language="ru", keep_puncs=True)
    print("`" + e.phonemize("hello how, are you today?") + "`")
