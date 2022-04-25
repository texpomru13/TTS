#!/usr/bin/python3
# -*- coding: utf-8 -*-
# OS: GNU/Linux, Author: Klim V. O.

'''
Предназначен для замены символов в тексте на специальные последовательности для корректного и красивого произношения
случайных буквенно-цифровых последовательностей. Например, 'МВУФ02ФОР-3/2703', 'ЛЮСЦЗЖМ-5/2503'.

Для полноты произношения модуль должен работать в паре с модулем расшифровки сокращений, в котором перечислены символы
наподобие '\\', '/', '*', '-', '_' и т.д.

Поддерживается только русский язык!
'''

import re
import time


# Словарь из пар "символ - специальная последовательность"
mapping_symbols_ru = {
    'а': 'аа+ъа',
    'б': 'бэ+',
    'в': 'вэ+',
    'г': 'гэ+',
    'д': 'дэ+',
    'е': 'е+',
    'ё': 'йо+',
    'ж': 'жэ+',
    'з': 'зэ+',
    'и': 'и+',
    'к': 'ка+',
    'л': 'э+л',
    'м': 'э+м',
    'н': 'э+н',
    'о': 'о+о',
    'п': 'пэ+',
    'р': 'э+р',
    'с': 'э+с',
    'т': 'тэ+',
    'у': 'уу+',
    'ф': 'э+ф',
    'х': 'ха+',
    'ц': 'цэ+',
    'ч': 'че+',
    'ш': 'ша+',
    'щ': 'ща+',
    'ъ': 'твёрдый знак',
    'ы': 'ыы+',
    'ь': 'мягкий знак',
    'э': 'ээ+',
    'ю': 'ю+',
    'я': 'я+',
    ' ': '. - ',

    # Список экранированных знаков препинания взят из TTS.tts.utils.text.phonemizers.text_preparation.expanding_abbreviations_ru
    ',': '\\,',
    '.': '\\.',
    ':': '\\:',
    ';': '\\;',
    '?': '\\?',
    '!': '\\!',
    '(': '\\(',
    ')': '\\)',
    '_': '\\_',
    '-': '\\-',
    '+': '\\+',
    '\'': '\\\'',
    '\\': '\\\\.'
}

framing_letters_with_spaces_re = re.compile(r'([^\s\d])')


def mapping_symbols_for_arbitrary_sequence(text, language='ru'):
    ''' Замена символов в тексте на специальные последовательности для корректного и красивого произношения случайных буквенно-цифровых последовательностей.
    Например, 'МВУФ02ФОР-3/2703', 'ЛЮСЦЗЖМ-5/2503'.

    Поддерживается весь русский алфавит, пробелы, знаки препинания ',.:;?!()_-+\'' экранируются символом '\\\\', все числа обрамляются пробелами и если
    какое-либо число содержит больше 2 цифр - добавление дефиса (неэкранированный символ '-') после каждой 2-ой цифры (так длинные числа будут легче
    восприниматься на слух).

    Для полноты произношения функция должна работать в паре с модулем расшифровки сокращений, в котором перечислены символы наподобие '\\\\', '/', '*',
    '-', '_' и т.д.

    1. text - строка с текстом для обработки
    2. language - язык текста, поддерживаются только русский язык 'ru'
    3. возвращает обработанный текст '''

    if language not in ['ru']:
        raise ValueError("Unsupported language: '{}', supported only 'ru' language.".format(language))

    if not text or text.isspace():
        return text

    # Обрамление каждого символа пробелами, кроме цифр и пробельных символов
    text = re.sub(framing_letters_with_spaces_re, r' \1 ', text)

    # Удаление нескольких подряд идущих пробелов и пробелов в начале и конце строки
    text = re.sub(r'\s{2,}', ' ', text)
    text = text.strip()

    # Приведение строки к нижнему регистру и конвертирование в список
    text = text.lower()
    text = list(text)

    # Если число содержит больше 2 цифр - добавление пробела после каждой 2-ой цифры (так длинные числа будут лучше восприниматься на слух)
    i = 0
    current_number_length = 0
    while i < len(text) - 1:
        if text[i].isdigit():
            current_number_length += 1
        else:
            current_number_length = 0
        if current_number_length == 2 and text[i+1].isdigit():
            text.insert(i+1, ' ')
            current_number_length = 0
        i += 1

    # Замена каждого символа в соответствии со словарём
    for i in range(len(text)-1, -1, -1):
        if mapping_symbols_ru.get(text[i]):
            text[i] = mapping_symbols_ru[text[i]]

    text = ''.join(text)
    return text


# Найденные баги:
# 1. Обратный слэш всегда воспринимается как символ экранирования, т.к. замена пробельного символа всегда начинается с точки (' ' -> '. - ') (частично
# исправлен)


def main():
    test_texts = [
        '',
        '  ',
        'ЁПТА чи+рик МВУФ02ФОР-3/2703 . (7820364.664) воробушек -+=(',
        'абвгдеёжзиклмнопрстуфхцчшщъыьэюя,.:;?!()_-+\\\''
    ]

    for text in test_texts:
        start_time = time.time()
        text = mapping_symbols_for_arbitrary_sequence(text, 'ru')
        elapsed_time = time.time() - start_time

        print("Результат: '{}', время обработки: {:.6f} сек".format(text, elapsed_time))


if __name__ == '__main__':
    main()
