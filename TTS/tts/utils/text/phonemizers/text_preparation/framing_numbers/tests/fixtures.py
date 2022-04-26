test_cases = [
    ("test 29а abc", "29а"),
    ("29а  ", "29а"),
    ("29а", "29а"),
    ("x29а", "x29а"),
    ("x29", "x29"),
    ("-39P", "-39P"),
    ("yx29", "yx29"),
    ("29aa", "29aa"),
    ("test29test", "29"),
    ("testx29", "29"),
    ("дом34", "34"),
    ("ст34", "ст34"),
    ("гелий3", "3"),
    ("гелий-3", "гелий-3"),
    ("3-гелий", "3-гелий"),
    ("водородий-3-гелий", "водородий-3-гелий"),
    ("или азот-3-йод аа", "азот-3-йод"),
    ("или фи-3-и аа", "фи-3-и"),
    ("водород", None),
    ("перерыв 12:00-13:00", "12:00-13:00"),
    ("P23А-34км", "P23А"),
    ("Д-4, корпус г1", "Д-4,"),
    ("345,90", "345,90"),
    ("-42,351.01", "-42,351.01"),
    ("42.356.342,..m103", "42.356.342,"),
    ("дома90-ых", "90-ых"),
    ("3,14здец", "3,14"),
    ("94т", "94т"),
    ("94т,", "94т,"),
    ("cтроение94т,", "94т,"),
    ("павильон 2В", "2В"),
    ("23проверка", "23"),
    ("а23проверка", "а23"),
    ("abc-def-35", "abc-def-35"),
    ("35-abc-def", "35-abc-def"),
    ("31ого", "31ого")
    ]


test_texts = {
    '': '',
    ' ': ' ',
    'Текст, не содержащий вообще какие-либо числа':
        'Текст, не содержащий вообще какие-либо числа',
    ' "А5", -39P\' "29Б", "Г-4", 452.4 "13-Е" д9 пом.12 гелий-3 номер6 87людей это 3,14здец, ':
        ' " А5 ", -39P \' " 29Б ", " Г-4 ", 452.4 " 13-Е " д9 пом. 12 гелий-3 номер 6 87 людей это 3,14 здец, ',
    'Вт-Сб 11:05-22:00, перерыв: 03:30-04:15':
        'Вт-Сб 11:05-22:00, перерыв: 03:30-04:15',
    '24,5+65=?&-+=(':
        '24,5 + 65 =?&-+=(',
    'Немного валют: 5$, 23.5$, 0,23$, -21,0€ +34€; минус83£ долг-45.32¥: 0¥-330₽=-7€ 34р и 0,9р':
        'Немного валют: 5$, 23.5$, 0,23$, -21,0€ + 34€; минус 83£ долг-45.32¥: 0¥ -330₽ = -7€ 34р и 0,9р',
    'По адр екб, пл.Чёрная, д5 корп.2, пом. 11, оф.9.':
        'По адр екб, пл.Чёрная, д5 корп. 2, пом. 11, оф. 9.',
    '2-3случая на (5-10)мин и 23000000000р и 210 501,51рубль?':
        '2-3 случая на (5-10) мин и 23000000000р и 210 501,51 рубль?',
    'дом 34, ст34, стр65, ст.34, стр.2, ст 4, стр 8, ст Новая, ст.Старая, стр. пять; д9, пом.18, оф12.':
        'дом 34, ст34, стр 65, ст. 34, стр. 2, ст 4, стр 8, ст Новая, ст.Старая, стр. пять; д9, пом. 18, оф12.',
    'ул. 25км, трасса Р-52 53км, трасса М2 43 км и P23А-34км':
        'ул. 25км, трасса Р-52 53км, трасса М2 43 км и P23А -34км',
    'М1 М-2, Р-153, Р89, дома 89а, Б23, 29Е, Д-4, корпус г1, строение94т, подвал32-в он шёл с 2 сумками, стр39 литера9 павильон2В':
        'М1 М-2, Р-153, Р89, дома 89а, Б23, 29Е, Д-4, корпус г1, строение 94т, подвал 32-в он шёл с 2 сумками, стр 39 литера 9 павильон 2В',
    '4-й переулок 20-летия победы, 9-е здание дома90-ых':
        '4-й переулок 20-летия победы, 9-е здание дома 90-ых',
    '12.03.2021, вчера было 11.03 - завтра будет 2021.03.13,дата:01-01-1961, и 29-04 после28:04':
        '12.03.2021, вчера было 11.03 - завтра будет 2021.03.13, дата: 01-01-1961, и 29-04 после 28:04',
    'Лечебно-Исправительное-Учреждение-10, мой 03дом на0 улице-201, уран-238, гелий-3,-98':
        'Лечебно-Исправительное-Учреждение-10, мой 03 дом на0 улице-201, уран-238, гелий-3, -98',
    'Medical-Correctiona-Institution-10, my 3-house is on03 09the street-201, uranium-238, helium-3,-98':
        'Medical-Correctiona-Institution-10, my 3-house is on03 09 the street-201, uranium-238, helium-3, -98',
    ' номер: -6 -811- 34-34 и -23 ребро  222,43  4,- о да! ':
        ' номер: -6 -811 - 34-34 и -23 ребро  222,43  4, - о да! ',
    '.м23,4-ыло-222,-=чпок-43-  *4-,-руб 50руб о да 56! (12 и 39) ,213, 32. 324, 234?32,05, 98:90:09, 9-05-2, 3-4, 43; 4;2!':
        '. м23,4 - ыло-222, -= чпок-43 -  * 4 -,-руб 50 руб о да 56! (12 и 39) , 213, 32. 324, 234 ? 32,05, 98:90:09, 9-05-2, 3-4, 43; 4; 2!',
    '12 345,90, -42,351.01, 42.356,34, 42,356,342, 42.356.342,..m103..,,,.400.uhv.,mn,,...':
        '12 345,90, -42,351.01, 42.356,34, 42,356,342, 42.356.342, .. m103. .,,,. 400. uhv.,mn,,...', }
