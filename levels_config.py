import flet as ft

VOWELS = set("АУОИІауоиі")

# Привітання Капітошки на головному екрані та реакція на закритий сірий замочок
WELCOME_SPEECH = (
    "Привіт, сонечко! Я — Капітошка! Радий бачити тебе у нашій країні літер. "
    "Обирай відкритий замочок на екрані та гайда гратися і вчитися читати разом!",
    "kapi_welcome.wav",
)

LOCKED_SPEECH = (
    "Ой, цей замочок ще спить! Тобі ще зарано тут займатися — обери відкритий замочок із зеленим значком!",
    "kapi_locked.wav",
)

# Точні тексти похвали Капітошки для правильних карток у всіх 10 замочках
CORRECT_PHRASES = {
    # Замочок 1 (А)
    "kapi_correct_avtobus.wav": "Правильно, молодець, автобус.",
    "kapi_correct_agrus.wav": "Правильно, молодець, аґрус.",
    "kapi_correct_ananas.wav": "Правильно, молодець, ананас.",
    # Замочок 2 (У)
    "kapi_correct_udav.wav": "Правильно, молодець, удав.",
    "kapi_correct_umyvalnyk.wav": "Правильно, молодець, умивальник.",
    "kapi_correct_usmishka.wav": "Правильно, молодець, усмішка.",
    # Замочок 3 (О)
    "kapi_correct_okulyary.wav": "Правильно, молодець! Окуляри!",
    "kapi_correct_oko.wav": "Правильно! Око! Зверни увагу: у слові о-ко аж дві букви о!",
    "kapi_correct_ogirok.wav": "Чудово! Огірок! Тут теж є дві букви о!",
    # Замочок 4 (М)
    "kapi_correct_mak.wav": "Правильно, молодець! Мак!",
    "kapi_correct_mysha.wav": "Правильно, молодець! Миша!",
    "kapi_correct_morozyvo.wav": "Правильно, морозиво!",
    # Замочок 5 (Н)
    "kapi_correct_nizh.wav": "Правильно, молодець! Ніж!",
    "kapi_correct_nozhytsi.wav": "Правильно! Ножиці!",
    "kapi_correct_nytka.wav": "Правильно, молодець! Нитка!",
    # Замочок 6 (И)
    "kapi_correct_mylo.wav": "Правильно, молодець! Мило!",
    "kapi_correct_lystok.wav": "Правильно! Листок!",
    "kapi_correct_lys.wav": "Правильно, молодець! Лис!",
    # Замочок 7 (І)
    "kapi_correct_indyk.wav": "Правильно, молодець! Індик!",
    "kapi_correct_igrashka.wav": "Правильно! Іграшка!",
    "kapi_correct_imbyr.wav": "Правильно, молодець! Імбир!",
    # Замочок 8 (С)
    "kapi_correct_slon.wav": "Правильно, молодець! Слон!",
    "kapi_correct_sontse.wav": "Правильно! Сонце!",
    "kapi_correct_sobaka.wav": "Правильно, молодець! Собака!",
    # Замочок 9 (Л)
    "kapi_correct_lev.wav": "Правильно, молодець! Лев!",
    "kapi_correct_lymon.wav": "Правильно! Лимон!",
    "kapi_correct_litak.wav": "Правильно, молодець! Літак!",
    # Замочок 10 (К)
    "kapi_correct_korova.wav": "Правильно, молодець! Корова!",
    "kapi_correct_kit.wav": "Правильно! Кіт!",
    "kapi_correct_kurka.wav": "Правильно, молодець! Курка!",
}
# Словник усіх ілюстрацій із папки assets/
WORD_IMAGES = {
    "Аґрус": "agrus.png", "Ананас": "ananas.png", "Автобус": "avtobus.png",
    "Будинок": "budynok.png", "Велосипед": "velosyped.png", "Дерево": "derevo.png",
    "Ракета": "raketa.png", "Удав": "udav.png", "Черепаха": "cherepaha.png",
    "Крісло": "krislo.png", "Диван": "dyvan.png", "Умивальник": "umyvalnyk.png",
    "Піца": "pizza.png", "Помідор": "pomidor.png", "Виноград": "vynograd.png",
    "Усмішка": "usmishka.png", "Цибуля": "zybulya.png",
    "Окуляри": "okulyary.png", "Равлик": "ravlyk.png", "Валіза": "valiza.png",
    "Риба": "ryba.png", "Кролик": "krolyk.png", "Око": "oko.png",
    "Собака": "sobaka.png", "Огірок": "ogirok.png", "Курка": "kurka.png",
    "Мак": "mak.png", "Жираф": "zhyraf.png", "Печиво": "pechyvo.png",
    "Пінгвін": "pingvin.png", "Двері": "dveri.png", "Миша": "mysha.png",
    "Шоколад": "shokolad.png", "Морозиво": "morozyvo.png", "Кокос": "kokos.png",
    "Павук": "pavuk.png", "Ніж": "nizh.png", "Телефон": "telefon.png",
    "Ножиці": "nozhytsi.png", "Оса": "osa.png", "Рюкзак": "ruksak.png",
    "Пилосос": "pylosos.png", "Хліб": "hlib.png", "Нитка": "nytka.png",
    "Сир": "syr.png", "Рушник": "rushnyk.png", "Мило": "mylo.png",
    "Корова": "korova.png", "Торт": "tort.png", "Листок": "lystok.png",
    "Кавун": "kavun.png", "Лис": "lysytsya.png", "Яйце": "yaytze.png",
    "Перець": "perets.png", "Гриб": "gryb.png", "Індик": "indyk.png",
    "Іграшка": "igrashka.png", "Коза": "koza.png", "Кукурудза": "kukurudza.png",
    "Білка": "bilka.png", "Гітара": "gitara.png", "Імбир": "imbyr.png",
    "Банан": "banan.png", "Лампа": "lampa.png", "Слон": "slon.png",
    "Сонце": "sontse.png", "Папуга": "papuga.png", "Кактус": "kaktus.png",
    "Гніздо": "gnizdo.png", "Подарунок": "podarunok.png",
    "Потяг": "potyag.png", "Лев": "lev.png", "Яблуко": "yabluko.png",
    "Лимон": "lymon.png", "Шапка": "shapka.png", "Чашка": "chashka.png",
    "Літак": "litak.png", "Олівці": "olivtsi.png", "Кіт": "kit.png",
    "Вишня": "vyshnya.png", "М’яч": "myach.png", "М'яч": "myach.png",
    "Подушка": "podushka.png",
}

LEVELS = {
    # ==========================================
    # ЗАМОЧОК 1: БУКВА Аа
    # ==========================================
    1: {
        "letter": "А",
        "face": "a",
        "color": ft.Colors.RED,
        "is_unlocked": True,
        "intro_big": (
            "Привіт! Поглянь на цю букву. Вона голосна, бо вміє дзвінко співати: А-а-а! Давай заспіваємо разом: А-а-а! Це — велика буква А. Вона висока, наче доросла! Великі букви завжди стоять на початку імен. Наприклад: Аліса, Артем, Андрійко. А ще вони відкривають кожне нове речення!",
            "kapi_intro_big.wav",
        ),
        "intro_small": (
            "А це — маленька буква а. Вона читається так само, як і велика: а-а-а!",
            "kapi_intro_small.wav",
        ),
        "wrong_game": ("Подумай ще! Обери слово, яке починається на літеру А.", "kapi_wrong_a.wav"),
        "try_again": ("Спробуй ще раз: А!", "kapi_try_again_a.wav"),
        "games": [
            {
                "prompt": "Розглянь малюнки! Обери слово, яке починається на літеру А!",
                "audio": "kapi_task1.wav",
                "items": [
                    ("Велосипед", False, ""),
                    ("Автобус", True, "kapi_correct_avtobus.wav"),
                    ("Дерево", False, ""),
                ],
            },
            {
                "prompt": "Розглянь малюнки! Обери слово, яке починається на літеру А!",
                "audio": "kapi_task2.wav",
                "items": [
                    ("Аґрус", True, "kapi_correct_agrus.wav"),
                    ("Будинок", False, ""),
                    ("Ракета", False, ""),
                ],
            },
            {
                "prompt": "Розглянь малюнки! Обери слово, яке починається на літеру А!",
                "audio": "kapi_task3.wav",
                "items": [
                    ("Око", False, ""),
                    ("Сонце", False, ""),
                    ("Ананас", True, "kapi_correct_ananas.wav"),
                ],
            },
        ],
        "reading_tasks": [
            {"display": "А", "target": "а", "face": "a", "audio": "kapi_read_step5_a.wav", "prompt": "Час читати! Веди пальчиком по екрану зліва направо. Прочитай велику букву. Зверни увагу, як широко відкривається ротик: А-а-а!"},
            {"display": "а", "target": "а", "face": "idle", "audio": "kapi_read_step6_a.wav", "prompt": "Молодець! А тепер прочитай маленьку букву."},
            {"display": "А-А-А", "target": "а", "face": "a", "audio": "kapi_read_step7_a.wav", "prompt": "Прочитай ці букви, веди пальчиком зліва направо."},
            {"display": "а-а-а", "target": "а", "face": "idle", "audio": "kapi_read_step8_a.wav", "prompt": "Тепер маленькі, теж зліва направо."},
            {"display": "А-а-а", "target": "а", "face": "a", "audio": "kapi_read_step9_a.wav", "prompt": "Згадаємо, які звуки називаються голосними? Так, їх можна співати! Проспівай і потягни."},
            {"display": "а", "target": "а", "face": "idle", "audio": "kapi_read_step10_a.wav", "prompt": "І наостанок: прочитай"},
            {"display": "А", "target": "а", "face": "a", "audio": "kapi_read_step11_a.wav", "prompt": "А тепер це"},
        ],
        "done": ("Чудова робота! Літеру Аа пройдено - новий рівень розблоковано!", "kapi_level_done.wav"),
    },

    # ==========================================
    # ЗАМОЧОК 2: БУКВА Уу
    # ==========================================
    2: {
        "letter": "У",
        "face": "u",
        "color": ft.Colors.RED,
        "is_unlocked": True,
        "intro_big": (
            "Привіт! Сьогодні ми вивчаємо букву У. Ця буква позначає голосний звук, як і літера А. Подивись, як ми складаємо губи дудочкою: У-у-у! Це велика літера У.",
            "kapi_intro_u_big.wav",
        ),
        "intro_small": (
            "А це маленька літера у. Губи теж дудочкою: у-у-у!",
            "kapi_intro_u_small.wav",
        ),
        "wrong_game": ("Подумай ще! Обери слово, яке починається на букву У.", "kapi_wrong_u.wav"),
        "games": [
            {
                "prompt": "Що зображено на малюнках? Знайди слово, яке починається на літеру У.",
                "audio": "kapi_find_u.wav",
                "items": [
                    ("Удав", True, "kapi_correct_udav.wav"),
                    ("Черепаха", False, ""),
                    ("Крісло", False, ""),
                ],
            },
            {
                "prompt": "Що зображено на малюнках? Знайди слово, яке починається на літеру У.",
                "audio": "kapi_find_u.wav",
                "items": [
                    ("Диван", False, ""),
                    ("Умивальник", True, "kapi_correct_umyvalnyk.wav"),
                    ("Піца", False, ""),
                ],
            },
            {
                "prompt": "Що зображено на малюнках? Знайди слово, яке починається на літеру У.",
                "audio": "kapi_find_u.wav",
                "items": [
                    ("Помідор", False, ""),
                    ("Виноград", False, ""),
                    ("Усмішка", True, "kapi_correct_usmishka.wav"),
                ],
            },
        ],
        "reading_tasks": [
            {"display": "У", "target": "у", "face": "u", "audio": "kapi_read_step5_u.wav", "prompt": "Прочитай велику букву."},
            {"display": "у", "target": "у", "face": "u", "audio": "kapi_read_step6_u.wav", "prompt": "Прочитай маленьку букву."},
            {"display": "У-у-у-у", "target": "у", "face": "u", "audio": "kapi_read_step7_u.wav", "prompt": "Пам'ятаєш, що ми читаємо зліва направо? Слідкуй за пальчиком!"},
            {"display": "а", "target": "а", "face": "a", "audio": "kapi_read_step8_u.wav", "prompt": "А тепер згадаємо попередню букву."},
            {"display": "У", "target": "у", "face": "u", "audio": "kapi_read_step9_u.wav", "prompt": "А що це за буква? Назви її!"},
            {"display": "А", "target": "а", "face": "a", "audio": "kapi_read_step10_u.wav", "prompt": "А цю пам'ятаєш?"},
            {"display": "у", "target": "у", "face": "u", "audio": "kapi_read_step11_u.wav", "prompt": "Давай прочитаємо цю маленьку літеру."},
            {"display": "У", "target": "у", "face": "u", "audio": "kapi_read_step12_u.wav", "prompt": "А тепер велика! Вимов гучно:"},
            {"display": "А", "target": "а", "face": "a", "audio": "kapi_read_step13_u.wav", "prompt": "Скажи, яка це буква?"},
            {"display": "а", "target": "а", "face": "a", "audio": "kapi_read_step14_u.wav", "prompt": "А цю зможеш прочитати?"},
            {"display": "У", "target": "у", "face": "u", "audio": "kapi_read_step15_u.wav", "prompt": "Увага, наступна літера!"},
            {"display": "а", "target": "а", "face": "a", "audio": "kapi_read_step16_u.wav", "prompt": "А це що за буква?"},
            {"display": "у", "target": "у", "face": "u", "audio": "kapi_read_step17_u.wav", "prompt": "Прочитай цю маленьку букву."},
            {"display": "А", "target": "а", "face": "a", "audio": "kapi_read_step18_u.wav", "prompt": "Назви цю велику літеру."},
            {"display": "У", "target": "у", "face": "u", "audio": "kapi_read_step19_u.wav", "prompt": "І знову ось ця літера!"},
            {"display": "Ау", "target": "ау", "face": "au", "audio": "kapi_read_step20_u.wav", "prompt": "Тепер найцікавіше! З'єднуємо букви у склади. Слідкуй за пальчиком і читай зліва направо: Ау"},
            {"display": "Уа", "target": "уа", "face": "ua", "audio": "kapi_read_step21_u.wav", "prompt": "А тепер навпаки: Уа"},
            {"display": "ау", "target": "ау", "face": "au", "audio": "kapi_read_step22_u.wav", "prompt": "Маленькі букви"},
            {"display": "уа", "target": "уа", "face": "ua", "audio": "kapi_read_step23_u.wav", "prompt": "І ще раз навпаки"},
            {"display": "У-а-а-а", "target": "уа", "alt_targets": ["а", "у", "ла"], "face": "ua", "image": "crying_kid.png", "audio": "kapi_read_step24_u.wav", "prompt": "Згадай, як плаче немовля"},
            {"display": "А-у-у-у", "target": "ау", "alt_targets": ["а", "у", "ла"], "face": "au", "image": "au.png", "audio": "kapi_read_step25_u.wav", "prompt": "А як гукає той, хто заблукав у лісі?"},
        ],
        "done": ("Ти справжній герой! Ми вивчили літеру Уу та навчилися читати перші склади!", "kapi_level_u_done.wav"),
    },

    # ==========================================
    # ЗАМОЧОК 3: БУКВА Оо
    # ==========================================
    3: {
        "letter": "О",
        "face": "o",
        "color": ft.Colors.RED,
        "is_unlocked": False,
        "intro_big": (
            "Привіт! Сьогодні вчимо букву Оо, яка позначає голосний звук! Склади губи у кружечок і повтори: О-о-о! Це велика літера О — кругла, як тарілка, м’яч чи годинник!",
            "kapi_intro_o_big.wav",
        ),
        "intro_small": (
            "А це маленька буква о. Знайди навколо себе круглі предмети, які нагадують цю літеру і проспівай разом зі мною: о-о-о!",
            "kapi_intro_o_small.wav",
        ),
        "wrong_game": ("Подумай ще! Обери слово, яке починається на букву О.", "kapi_wrong_o.wav"),
        "games": [
            {
                "prompt": "Що зображено на малюнках? Обери слово, яке починається на літеру О!",
                "audio": "kapi_find_o.wav",
                "items": [
                    ("Окуляри", True, "kapi_correct_okulyary.wav"),
                    ("Равлик", False, ""),
                    ("Валіза", False, ""),
                ],
            },
            {
                "prompt": "Що зображено на малюнках? Обери слово, яке починається на літеру О!",
                "audio": "kapi_find_o.wav",
                "items": [
                    ("Риба", False, ""),
                    ("Кролик", False, ""),
                    ("Око", True, "kapi_correct_oko.wav"),
                ],
            },
            {
                "prompt": "Що зображено на малюнках? Обери слово, яке починається на літеру О!",
                "audio": "kapi_find_o.wav",
                "items": [
                    ("Собака", False, ""),
                    ("Огірок", True, "kapi_correct_ogirok.wav"),
                    ("Курка", False, ""),
                ],
            },
        ],
        "reading_tasks": [
            {"display": "О-о-о!", "target": "о", "face": "o", "audio": "kapi_read_step9_a.wav", "prompt": "Згадаємо, які звуки називаються голосними? Так, їх можна співати! Проспівай і потягни."},
            {"display": "А", "target": "а", "face": "a", "audio": "kapi_read_step5_u.wav", "prompt": "Прочитай велику букву."},
            {"display": "о", "target": "о", "face": "o", "audio": "kapi_read_step6_u.wav", "prompt": "Прочитай маленьку букву."},
            {"display": "О", "target": "о", "face": "o", "audio": "kapi_read_step12_u.wav", "prompt": "А тепер велика! Вимов гучно:"},
            {"display": "у", "target": "у", "face": "u", "audio": "kapi_read_step11_u.wav", "prompt": "Давай прочитаємо цю маленьку літеру."},
            {"display": "А", "target": "а", "face": "a", "audio": "kapi_read_step10_u.wav", "prompt": "А цю пам'ятаєш?"},
            {"display": "У", "target": "у", "face": "u", "audio": "kapi_read_step18_u.wav", "prompt": "Назви цю велику літеру."},
            {"display": "а", "target": "а", "face": "a", "audio": "kapi_read_step14_u.wav", "prompt": "А цю зможеш прочитати?"},
            {"display": "о", "target": "о", "face": "o", "audio": "kapi_read_step17_u.wav", "prompt": "Прочитай цю маленьку букву."},
            {"display": "У", "target": "у", "face": "u", "audio": "kapi_read_step15_u.wav", "prompt": "Увага, наступна літера!"},
            {"display": "а", "target": "а", "face": "a", "audio": "kapi_read_step16_u.wav", "prompt": "А це що за буква?"},
            {"display": "О", "target": "о", "face": "o", "audio": "kapi_read_step13_u.wav", "prompt": "Скажи, яка це буква?"},
            {"display": "у", "target": "у", "face": "u", "audio": "kapi_read_step19_u.wav", "prompt": "І знову ось ця літера!"},
            {"display": "АО", "target": "ао", "face": "o", "audio": "kapi_read_o_merge.wav", "prompt": "Тепер з’єднуємо букви! Тягни першу літеру голосом і плавно переходь у другу, щоб звучало разом: АО!"},
            {"display": "УА", "target": "уа", "face": "ua", "audio": "kapi_read_step21_u.wav", "prompt": "А тепер навпаки: Уа"},
            {"display": "ау", "target": "ау", "face": "au", "audio": "kapi_drill_syllable_1.wav", "prompt": "З’єднай букви плавно, на одному подиху!"},
        ],
        "done": ("Ура! Ми вивчили літеру Оо та навчилися плавно з’єднувати голосні!", "kapi_level_o_done.wav"),
    },

    # ==========================================
    # ЗАМОЧОК 4: БУКВА Мм
    # ==========================================
    4: {
        "letter": "М",
        "face": "m",
        "color": ft.Colors.BLUE,
        "is_unlocked": False,
        "intro_big": (
            "Знайомся - це літера Мм! Буква синього кольору, бо позначає приголосний звук. Кажемо коротко, стискаючи губи: М! Це велика буква М.",
            "kapi_intro_m_big.wav",
        ),
        "intro_small": (
            "А це маленька буква м. Червоні голосні літери дзвінко співають, а сині приголосні самі не тягнуться, зате разом із червоними складають слова!",
            "kapi_intro_m_small.wav",
        ),
        "wrong_game": ("Подумай ще! Обери слово, яке починається на букву М.", "kapi_wrong_m.wav"),
        "games": [
            {
                "prompt": "Розглянь малюнки! Обери слово, яке починається на літеру М!",
                "audio": "kapi_find_m.wav",
                "items": [
                    ("Мак", True, "kapi_correct_mak.wav"),
                    ("Жираф", False, ""),
                    ("Печиво", False, ""),
                ],
            },
            {
                "prompt": "Розглянь малюнки! Обери слово, яке починається на літеру М!",
                "audio": "kapi_find_m.wav",
                "items": [
                    ("Пінгвін", False, ""),
                    ("Двері", False, ""),
                    ("Миша", True, "kapi_correct_mysha.wav"),
                ],
            },
            {
                "prompt": "Розглянь малюнки! Обери слово, яке починається на літеру М!",
                "audio": "kapi_find_m.wav",
                "items": [
                    ("Шоколад", False, ""),
                    ("Морозиво", True, "kapi_correct_morozyvo.wav"),
                    ("Кокос", False, ""),
                ],
            },
        ],
        "reading_tasks": [
            {"display": "АМ", "target": "ам", "face": "m", "audio": "kapi_read_m_closed.wav", "prompt": "Читаємо закритий склад! Тягни першу червону літеру і плавно приєднуй М без паузи: ам!"},
            {"display": "ам", "target": "ам", "face": "m", "audio": "kapi_read_step22_u.wav", "prompt": "Маленькі букви"},
            {"display": "УМ", "target": "ум", "face": "m", "audio": "kapi_drill_syllable_1.wav", "prompt": "З’єднай букви плавно, на одному подиху!"},
            {"display": "ум", "target": "ум", "face": "m", "audio": "kapi_read_step22_u.wav", "prompt": "Маленькі букви"},
            {"display": "ОМ", "target": "ом", "face": "m", "audio": "kapi_drill_syllable_2.wav", "prompt": "А що вийде тут? Прочитай разом, без паузи!"},
            {"display": "ом", "target": "ом", "face": "m", "audio": "kapi_read_step11_a.wav", "prompt": "А тепер це"},
            {"display": "МУ", "target": "му", "face": "u", "audio": "kapi_read_m_open.wav", "prompt": "Тепер відкритий склад! Не додавай нічого між буквами: скажи М і одразу відкрий ротик на У — МУ!"},
            {"display": "му", "target": "му", "face": "u", "audio": "kapi_read_step22_u.wav", "prompt": "Маленькі букви"},
            {"display": "Му-му-му!", "target": "мумуму", "alt_targets": ["муму"], "face": "u", "image": "cow_mu.png", "audio": "kapi_read_m_cow.wav", "prompt": "А як мукає корівка? Прочитай разом!"},
            {"display": "Ам-ам!", "target": "амам", "face": "m", "image": "am-am.png", "audio": "kapi_read_m_amam.wav", "prompt": "Прочитай, як ми кажемо, коли їмо щось смачненьке!"},
            {"display": "ма-ма", "target": "мама", "face": "a", "image": "mama.png", "audio": "kapi_read_m_mama.wav", "prompt": "Час для нашого першого справжнього слова! З’єднай два склади разом!"},
            {"display": "ма-мо", "target": "мамо", "face": "o", "image": "mama.png", "audio": "kapi_drill_word_1.wav", "prompt": "З’єднай склади і прочитай ціле слово!"},
            {"display": "ма-му", "target": "маму", "face": "u", "image": "mama.png", "audio": "kapi_read_step10_a.wav", "prompt": "І наостанок: прочитай"},
        ],
        "done": ("Ти супер! Тепер ти вмієш читати слова!", "kapi_level_m_done.wav"),
    },

    # ==========================================
    # ЗАМОЧОК 5: БУКВА Нн
    # ==========================================
    5: {
        "letter": "Н",
        "face": "idle",
        "color": ft.Colors.BLUE,
        "is_unlocked": False,
        "intro_big": (
            "Це - велика синя літера, яка позначає приголосний звук Н! Вимовляй чисто й коротко: Н!",
            "kapi_intro_n_big.wav",
        ),
        "intro_small": (
            "А це маленька літера н. Сьогодні ми прочитаємо багато складів і наше перше речення!",
            "kapi_intro_n_small.wav",
        ),
        "wrong_game": ("Подумай ще! Шукай слово на букву Н.", "kapi_wrong_n.wav"),
        "games": [
            {
                "prompt": "Обери слово, яке починається на літеру Н!",
                "audio": "kapi_find_n.wav",
                "items": [
                    ("Павук", False, ""),
                    ("Ніж", True, "kapi_correct_nizh.wav"),
                    ("Телефон", False, ""),
                ],
            },
            {
                "prompt": "Обери слово, яке починається на літеру Н!",
                "audio": "kapi_find_n.wav",
                "items": [
                    ("Ножиці", True, "kapi_correct_nozhytsi.wav"),
                    ("Оса", False, ""),
                    ("Рюкзак", False, ""),
                ],
            },
            {
                "prompt": "Обери слово, яке починається на літеру Н!",
                "audio": "kapi_find_n.wav",
                "items": [
                    ("Пилосос", False, ""),
                    ("Хліб", False, ""),
                    ("Нитка", True, "kapi_correct_nytka.wav"),
                ],
            },
        ],
        "reading_tasks": [
            {"display": "АН", "target": "ан", "audio": "kapi_read_n_closed.wav", "prompt": "Починаємо із закритого складу: тягни перший голосний і плавно додавай н!"},
            {"display": "ан", "target": "ан", "audio": "kapi_read_step22_u.wav", "prompt": "Маленькі букви"},
            {"display": "ОН", "target": "он", "audio": "kapi_drill_syllable_1.wav", "prompt": "З’єднай букви плавно, на одному подиху!"},
            {"display": "он", "target": "он", "audio": "kapi_read_step22_u.wav", "prompt": "Маленькі букви"},
            {"display": "УН", "target": "ун", "audio": "kapi_drill_syllable_2.wav", "prompt": "А що вийде тут? Прочитай разом, без паузи!"},
            {"display": "ун", "target": "ун", "audio": "kapi_read_step22_u.wav", "prompt": "Маленькі букви"},
            {"display": "НА", "target": "на", "audio": "kapi_read_n_open.wav", "prompt": "А тепер відкритий склад: зі звуку н одразу переходь у голосний!"},
            {"display": "на", "target": "на", "audio": "kapi_read_step22_u.wav", "prompt": "Маленькі букви"},
            {"display": "ам", "target": "ам", "audio": "kapi_drill_careful.wav", "prompt": "Читай уважно кожну букву, не вгадуй!"},
            {"display": "ом", "target": "ом", "audio": "kapi_drill_syllable_1.wav", "prompt": "З’єднай букви плавно, на одному подиху!"},
            {"display": "му", "target": "му", "audio": "kapi_drill_syllable_2.wav", "prompt": "А що вийде тут? Прочитай разом, без паузи!"},
            {"display": "ма", "target": "ма", "audio": "kapi_read_step11_a.wav", "prompt": "А тепер це"},
            {"display": "мо", "target": "мо", "audio": "kapi_drill_careful.wav", "prompt": "Читай уважно кожну букву, не вгадуй!"},
            {"display": "ну", "target": "ну", "audio": "kapi_drill_syllable_1.wav", "prompt": "З’єднай букви плавно, на одному подиху!"},
            {"display": "но", "target": "но", "audio": "kapi_drill_syllable_2.wav", "prompt": "А що вийде тут? Прочитай разом, без паузи!"},
            {
                "display": "На, мамо,{kvity.png}.",
                "target": "намамоквіти",
                "image": "kvity.png",
                "audio": "kapi_read_n_sentence.wav",
                "prompt": "Наше перше речення! Воно починається з великої букви, а в кінці стоїть крапка. Останнє слово в цьому реченні позначено картинкою!",
            },
        ],
        "done": ("Браво! Ти вже читаєш ціле речення!", "kapi_level_n_done.wav"),
    },

    # ==========================================
    # ЗАМОЧОК 6: БУКВА Ии
    # ==========================================
    6: {
        "letter": "И",
        "face": "y",
        "color": ft.Colors.RED,
        "is_unlocked": False,
        "intro_big": (
            "Познайомся з буквою Ии, яка позначає голосний звук! Велика буква майже не вживається, бо є дуже мало слів, які з неї починаються.",
            "kapi_intro_y_big.wav",
        ),
        "intro_small": (
            "Подивись на слова сир та рушник: маленька червона буква и ховається всередині слів!",
            "kapi_intro_y_small.wav",
        ),
        "demo_words": [("Сир", "syr.png"), ("Рушник", "rushnyk.png")],
        "wrong_game": ("Вимов слова ще раз! У якому слові є буква и?", "kapi_wrong_y.wav"),
        "games": [
            {
                "prompt": "А тепер самостійно обери слово, всередині якого заховалася буква и!",
                "audio": "kapi_find_y.wav",
                "items": [
                    ("Мило", True, "kapi_correct_mylo.wav"),
                    ("Корова", False, ""),
                    ("Торт", False, ""),
                ],
            },
            {
                "prompt": "А тепер самостійно обери слово, всередині якого заховалася буква и!",
                "audio": "kapi_find_y.wav",
                "items": [
                    ("Ананас", False, ""),
                    ("Листок", True, "kapi_correct_lystok.wav"),
                    ("Кавун", False, ""),
                ],
            },
            {
                "prompt": "А тепер самостійно обери слово, всередині якого заховалася буква и!",
                "audio": "kapi_find_y.wav",
                "items": [
                    ("Лис", True, "kapi_correct_lys.wav"),
                    ("Кокос", False, ""),
                    ("Яйце", False, ""),
                ],
            },
        ],
        "reading_tasks": [
            {"display": "И-и-и-и!", "target": "и", "face": "y", "image": "crying_baby.png", "audio": "kapi_read_step24_u.wav", "prompt": "Згадай, як плаче немовля"},
            {"display": "ни", "target": "ни", "face": "y", "audio": "kapi_drill_careful.wav", "prompt": "Читай уважно кожну букву, не вгадуй!"},
            {"display": "на", "target": "на", "face": "a", "audio": "kapi_drill_syllable_1.wav", "prompt": "З’єднай букви плавно, на одному подиху!"},
            {"display": "ну", "target": "ну", "face": "u", "audio": "kapi_drill_syllable_2.wav", "prompt": "А що вийде тут? Прочитай разом, без паузи!"},
            {"display": "ми", "target": "ми", "face": "y", "audio": "kapi_drill_careful.wav", "prompt": "Читай уважно кожну букву, не вгадуй!"},
            {"display": "ма", "target": "ма", "face": "a", "audio": "kapi_read_step11_a.wav", "prompt": "А тепер це"},
            {"display": "мо", "target": "мо", "face": "o", "audio": "kapi_drill_syllable_1.wav", "prompt": "З’єднай букви плавно, на одному подиху!"},
            {"display": "ум", "target": "ум", "face": "m", "audio": "kapi_drill_syllable_2.wav", "prompt": "А що вийде тут? Прочитай разом, без паузи!"},
            {"display": "ун", "target": "ун", "audio": "kapi_drill_careful.wav", "prompt": "Читай уважно кожну букву, не вгадуй!"},
            {"display": "У  мами  ми.", "target": "умамими", "image": "u_mamy_my.png", "audio": "kapi_drill_sentence_1.wav", "prompt": "Прочитай речення від великої букви до крапки!"},
            {"display": "ам-ам", "target": "амам", "face": "m", "image": "am-am.png", "audio": "kapi_read_m_amam.wav", "prompt": "Прочитай, як ми кажемо, коли їмо щось смачненьке!"},
            {"display": "ма-ма", "target": "мама", "face": "a", "image": "mama.png", "audio": "kapi_drill_word_1.wav", "prompt": "З’єднай склади і прочитай ціле слово!"},
            {"display": "му-му", "target": "муму", "face": "u", "image": "korova.png", "audio": "kapi_read_m_cow.wav", "prompt": "А як мукає корівка? Прочитай разом!"},
        ],
        "done": ("Чудова робота! Буква и тобі підкорилася!", "kapi_level_y_done.wav"),
    },

    # ==========================================
    # ЗАМОЧОК 7: БУКВА Іі
    # ==========================================
    7: {
        "letter": "І",
        "face": "i",
        "color": ft.Colors.RED,
        "is_unlocked": False,
        "intro_big": (
            "Це велика червона буква І, схожа на свічку! Вона позначає ніжний голосний звук: І-і-і!",
            "kapi_intro_i_big.wav",
        ),
        "intro_small": (
            "А це маленька буква і з крапочкою. Вона робить приголосні звуки перед собою м’якими та лагідними!",
            "kapi_intro_i_small.wav",
        ),
        "wrong_game": ("Подумай ще! Обери слово, що починається на букву І.", "kapi_wrong_i.wav"),
        "games": [
            {
                "prompt": "Обери слово, яке починається на літеру І!",
                "audio": "kapi_find_i.wav",
                "items": [
                    ("Перець", False, ""),
                    ("Гриб", False, ""),
                    ("Індик", True, "kapi_correct_indyk.wav"),
                ],
            },
            {
                "prompt": "Обери слово, яке починається на літеру І!",
                "audio": "kapi_find_i.wav",
                "items": [
                    ("Іграшка", True, "kapi_correct_igrashka.wav"),
                    ("Коза", False, ""),
                    ("Кукурудза", False, ""),
                ],
            },
            {
                "prompt": "Обери слово, яке починається на літеру І!",
                "audio": "kapi_find_i.wav",
                "items": [
                    ("Білка", False, ""),
                    ("Гітара", False, ""),
                    ("Імбир", True, "kapi_correct_imbyr.wav"),
                ],
            },
        ],
        "reading_tasks": [
            {"display": "им", "target": "им", "face": "m", "audio": "kapi_read_i_soft.wav", "prompt": "Давай порівняємо тверді та м’які склади! Читай плавно і слухай, як змінюється звучання."},
            {"display": "ім", "target": "ім", "face": "i", "audio": "kapi_drill_syllable_1.wav", "prompt": "З’єднай букви плавно, на одному подиху!"},
            {"display": "ин", "target": "ин", "face": "y", "audio": "kapi_drill_careful.wav", "prompt": "Читай уважно кожну букву, не вгадуй!"},
            {"display": "ін", "target": "ін", "face": "i", "audio": "kapi_drill_syllable_2.wav", "prompt": "А що вийде тут? Прочитай разом, без паузи!"},
            {"display": "ни", "target": "ни", "face": "y", "audio": "kapi_drill_careful.wav", "prompt": "Читай уважно кожну букву, не вгадуй!"},
            {"display": "ні", "target": "ні", "face": "i", "audio": "kapi_read_step11_a.wav", "prompt": "А тепер це"},
            {"display": "ми", "target": "ми", "face": "y", "audio": "kapi_drill_syllable_1.wav", "prompt": "З’єднай букви плавно, на одному подиху!"},
            {"display": "мі", "target": "мі", "face": "i", "audio": "kapi_drill_syllable_2.wav", "prompt": "А що вийде тут? Прочитай разом, без паузи!"},
            {"display": "на-ми", "target": "нами", "audio": "kapi_drill_word_1.wav", "prompt": "З’єднай склади і прочитай ціле слово!"},
            {"display": "ни-ми", "target": "ними", "audio": "kapi_drill_word_2.wav", "prompt": "А яке слово заховалося тут? Прочитай плавно!"},
            {"display": "ма-мі", "target": "мамі", "face": "i", "audio": "kapi_drill_word_1.wav", "prompt": "З’єднай склади і прочитай ціле слово!"},
            {"display": "Ні-на", "target": "ніна", "audio": "kapi_read_i_names.wav", "prompt": "Імена завжди пишемо з великої літери! Прочитай ім’я по складах."},
            {"display": "Ін-на", "target": "інна", "audio": "kapi_drill_word_2.wav", "prompt": "А яке слово заховалося тут? Прочитай плавно!"},
            {"display": "Он  мама  і  Ні-на.", "target": "онмамаініна", "image": "mama_i_nina.png", "audio": "kapi_drill_sentence_1.wav", "prompt": "Прочитай речення від великої букви до крапки!"},
            {"display": "У  мами {apples.png}.", "target": "умамияблука", "audio": "kapi_read_i_apples1.wav", "prompt": "Прочитай речення, а замість малюнка скажи правильне слово!"},
            {"display": "На,  Ніно, {apples.png}.", "target": "нанінояблука", "audio": "kapi_read_i_apples2.wav", "prompt": "Прочитай наступне речення і теж додай слово з малюнка!"},
        ],
        "done": ("Молодець! Рівень букви І успішно пройдено!", "kapi_level_i_done.wav"),
    },

    # ==========================================
    # ЗАМОЧОК 8: БУКВА Сс
    # ==========================================
    8: {
        "letter": "С",
        "face": "idle",
        "color": ft.Colors.BLUE,
        "is_unlocked": False,
        "intro_big": (
            "Це велика синя буква, що позначає приголосний звук С! Вимовляй її як тихий вітерець, без зайвих звуків: С!",
            "kapi_intro_s_big.wav",
        ),
        "intro_small": (
            "А це маленька буква с. Зараз ми навчимося читати слова з трьох літер за допомогою лінії!",
            "kapi_intro_s_small.wav",
        ),
        "wrong_game": ("Подумай ще! Шукай слово на букву С.", "kapi_wrong_s.wav"),
        "games": [
            {
                "prompt": "Обери слово, яке починається на літеру С!",
                "audio": "kapi_find_s.wav",
                "items": [
                    ("Банан", False, ""),
                    ("Лампа", False, ""),
                    ("Слон", True, "kapi_correct_slon.wav"),
                ],
            },
            {
                "prompt": "Обери слово, яке починається на літеру С!",
                "audio": "kapi_find_s.wav",
                "items": [
                    ("Сонце", True, "kapi_correct_sontse.wav"),
                    ("Папуга", False, ""),
                    ("Кактус", False, ""),
                ],
            },
            {
                "prompt": "Обери слово, яке починається на літеру С!",
                "audio": "kapi_find_s.wav",
                "items": [
                    ("Гніздо", False, ""),
                    ("Собака", True, "kapi_correct_sobaka.wav"),
                    ("Подарунок", False, ""),
                ],
            },
        ],
        "reading_tasks": [
            {"display": "ми", "target": "ми", "audio": "kapi_drill_syllable_1.wav", "prompt": "З’єднай букви плавно, на одному подиху!"},
            {"display": "на┆с", "target": "нас", "audio": "kapi_drill_line.wav", "prompt": "Зверни увагу! Спочатку прочитай частинку до лінії, і одразу додай наступну букву!"},
            {"display": "на┆м", "target": "нам", "audio": "kapi_drill_line.wav", "prompt": "Зверни увагу! Спочатку прочитай частинку до лінії, і одразу додай наступну букву!"},
            {"display": "у на┆с", "target": "унас", "audio": "kapi_drill_word_2.wav", "prompt": "А яке слово заховалося тут? Прочитай плавно!"},
            {"display": "си┆н", "target": "син", "audio": "kapi_drill_line.wav", "prompt": "Зверни увагу! Спочатку прочитай частинку до лінії, і одразу додай наступну букву!"},
            {"display": "си-ни", "target": "сини", "audio": "kapi_drill_word_1.wav", "prompt": "З’єднай склади і прочитай ціле слово!"},
            {"display": "со┆м", "target": "сом", "image": "som.png", "audio": "kapi_drill_line.wav", "prompt": "Зверни увагу! Спочатку прочитай частинку до лінії, і одразу додай наступну букву!"},
            {"display": "со-ми", "target": "соми", "audio": "kapi_drill_word_2.wav", "prompt": "А яке слово заховалося тут? Прочитай плавно!"},
            {"display": "со┆с-на", "target": "сосна", "image": "sosna.png", "audio": "kapi_drill_word_1.wav", "prompt": "З’єднай склади і прочитай ціле слово!"},
            {"display": "на-со┆с", "target": "насос", "image": "nasos.png", "audio": "kapi_drill_word_2.wav", "prompt": "А яке слово заховалося тут? Прочитай плавно!"},
            {"display": "У  ма-ми  син.  У  си-на  сом.", "target": "умамисинусинасом", "audio": "kapi_read_s_sentences.wav", "prompt": "Прочитай ці речення так, щоб добре зрозуміти, хто є у мами, а що є у сина!"},
            {"display": "Ма-мо!  О-си!", "target": "мамооси", "image": "osa.png", "audio": "kapi_read_s_exclamation.wav", "prompt": "У кінці цього речення замість крапки стоїть знак оклику! Це означає, що це речення треба прочитати голосно та емоційно!"},
        ],
        "done": ("Супер! Ти вже легко читаєш слова та речення!", "kapi_level_s_done.wav"),
    },

    # ==========================================
    # ЗАМОЧОК 9: БУКВА Лл
    # ==========================================
    9: {
        "letter": "Л",
        "face": "idle",
        "color": ft.Colors.BLUE,
        "is_unlocked": False,
        "intro_big": (
            "Вітаю на новому рівні! Це велика буква Л. Торкнися кінчиком язика верхніх зубів і скажи чистий звук: Л!",
            "kapi_intro_l_big.wav",
        ),
        "intro_small": (
            "А це маленька буква л.",
            "kapi_intro_l_small.wav",
        ),
        "wrong_game": ("Подумай ще! Обери слово, яке починається на букву Л.", "kapi_wrong_l.wav"),
        "games": [
            {
                "prompt": "Обери слово, яке починається на літеру Л!",
                "audio": "kapi_find_l.wav",
                "items": [
                    ("Потяг", False, ""),
                    ("Лев", True, "kapi_correct_lev.wav"),
                    ("Яблуко", False, ""),
                ],
            },
            {
                "prompt": "Обери слово, яке починається на літеру Л!",
                "audio": "kapi_find_l.wav",
                "items": [
                    ("Лимон", True, "kapi_correct_lymon.wav"),
                    ("Слон", False, ""),
                    ("Кавун", False, ""),
                ],
            },
            {
                "prompt": "Обери слово, яке починається на літеру Л!",
                "audio": "kapi_find_l.wav",
                "items": [
                    ("Шапка", False, ""),
                    ("Чашка", False, ""),
                    ("Літак", True, "kapi_correct_litak.wav"),
                ],
            },
        ],
        "reading_tasks": [
            {"display": "лі┆с", "target": "ліс", "image": "lis.png", "audio": "kapi_drill_line.wav", "prompt": "Зверни увагу! Спочатку прочитай частинку до лінії, і одразу додай наступну букву!"},
            {"display": "ли┆с", "target": "лис", "image": "lysytsya.png", "audio": "kapi_read_l_change.wav", "prompt": "Зверни увагу, як заміна лише однієї букви перетворила ліс на зовсім інше слово! Прочитай!"},
            {"display": "ми-ло", "target": "мило", "image": "mylo.png", "audio": "kapi_drill_word_1.wav", "prompt": "З’єднай склади і прочитай ціле слово!"},
            {"display": "ло-ми", "target": "ломи", "audio": "kapi_read_step23_u.wav", "prompt": "І ще раз навпаки"},
            {"display": "ма-ло", "target": "мало", "audio": "kapi_drill_word_2.wav", "prompt": "А яке слово заховалося тут? Прочитай плавно!"},
            {"display": "ма┆с-ло", "target": "масло", "image": "maslo.png", "audio": "kapi_drill_line.wav", "prompt": "Зверни увагу! Спочатку прочитай частинку до лінії, і одразу додай наступну букву!"},
            {"display": "с┆мо-ла", "target": "смола", "audio": "kapi_drill_line.wav", "prompt": "Зверни увагу! Спочатку прочитай частинку до лінії, і одразу додай наступну букву!"},
            {"display": "ли-мо-ни", "target": "лимони", "image": "lymon.png", "audio": "kapi_drill_word_1.wav", "prompt": "З’єднай склади і прочитай ціле слово!"},
            {"display": "Ма-ма  ми-ла  Лі-ну.  У  ма-ми  ми-ло.", "target": "мамамилалінуумамимило", "audio": "kapi_read_l_lina.wav", "prompt": "Прочитай два речення і скажи: кого мила мама?"},
            {"display": "Ліс.  Лі-на  і  Ні-на  у  лі-сі.", "target": "ліслінаінінаулісі", "image": "lis.png", "audio": "kapi_read_l_story1.wav", "prompt": "Читаємо справжнє оповідання про ліс! Прочитай назву і перші речення."},
            {"display": "У  сос-ни {shyshky.png}  і  {golky.png}.", "target": "усоснишишкиіголки", "audio": "kapi_read_l_story2.wav", "prompt": "Завершуємо оповідання! Прочитай початок і заміни малюнки відповідними словами!", "finger_offset_y": 78},
        ],
        "done": ("Неймовірно! Ти прочитав справжнє оповідання про ліс!", "kapi_level_l_done.wav"),
    },

    # ==========================================
    # ЗАМОЧОК 10: БУКВА Кк
    # ==========================================
    10: {
        "letter": "К",
        "face": "idle",
        "color": ft.Colors.BLUE,
        "is_unlocked": False,
        "intro_big": (
            "Знайомся! Ця велика буква позначає приголосний звук К. Вимовляй коротко й чітко: К!",
            "kapi_intro_k_big.wav",
        ),
        "intro_small": (
            "А це маленька буква к. Зараз ми підемо від простих складів до великих цікавих речень!",
            "kapi_intro_k_small.wav",
        ),
        "wrong_game": ("Подумай ще! Шукай слово на літеру К.", "kapi_wrong_k.wav"),
        "games": [
            {
                "prompt": "Обери слово, яке починається на букву К!",
                "audio": "kapi_find_k.wav",
                "items": [
                    ("Яйце", False, ""),
                    ("Корова", True, "kapi_correct_korova.wav"),
                    ("Олівці", False, ""),
                ],
            },
            {
                "prompt": "Обери слово, яке починається на букву К!",
                "audio": "kapi_find_k.wav",
                "items": [
                    ("Кіт", True, "kapi_correct_kit.wav"),
                    ("Вишня", False, ""),
                    ("Торт", False, ""),
                ],
            },
            {
                "prompt": "Обери слово, яке починається на букву К!",
                "audio": "kapi_find_k.wav",
                "items": [
                    ("М’яч", False, ""),
                    ("Подушка", False, ""),
                    ("Курка", True, "kapi_correct_kurka.wav"),
                ],
            },
        ],
        "reading_tasks": [
            {"display": "ка", "target": "ка", "audio": "kapi_drill_syllable_1.wav", "prompt": "З’єднай букви плавно, на одному подиху!"},
            {"display": "ку", "target": "ку", "audio": "kapi_drill_syllable_2.wav", "prompt": "А що вийде тут? Прочитай разом, без паузи!"},
            {"display": "ко", "target": "ко", "audio": "kapi_drill_careful.wav", "prompt": "Читай уважно кожну букву, не вгадуй!"},
            {"display": "ки", "target": "ки", "audio": "kapi_read_step11_a.wav", "prompt": "А тепер це"},
            {"display": "ко-ло", "target": "коло", "audio": "kapi_drill_word_1.wav", "prompt": "З’єднай склади і прочитай ціле слово!"},
            {"display": "ко-ло┆с", "target": "колос", "image": "kolos.png", "audio": "kapi_drill_line.wav", "prompt": "Зверни увагу! Спочатку прочитай частинку до лінії, і одразу додай наступну букву!"},
            {"display": "ма┆к", "target": "мак", "image": "mak.png", "audio": "kapi_drill_line.wav", "prompt": "Зверни увагу! Спочатку прочитай частинку до лінії, і одразу додай наступну букву!"},
            {"display": "с┆ма┆к", "target": "смак", "audio": "kapi_drill_word_2.wav", "prompt": "А яке слово заховалося тут? Прочитай плавно!"},
            {"display": "о-са", "target": "оса", "image": "osa.png", "audio": "kapi_drill_word_1.wav", "prompt": "З’єднай склади і прочитай ціле слово!"},
            {"display": "ко-са", "target": "коса", "image": "kosa.png", "audio": "kapi_drill_word_2.wav", "prompt": "А яке слово заховалося тут? Прочитай плавно!"},
            {"display": "Ко-ло  к┆лу-ні  ма-ли-на.  У  Ні-ни {koshyk.png}.", "target": "колоклунімалинауніникошик", "audio": "kapi_read_k_koshyk.wav", "prompt": "Прочитай речення про малину і Ніну, а в кінці назви відповідне слово із малюнка!", "finger_offset_y": 78},
            {
                "display": "У  ма-ми  мо-ло-ко.  На,  Лі-но,  ка-ка-о.",
                "target": "умамимолоконалінокакао",
                "audio": "kapi_read_k_final.wav",
                "prompt": "І фінальні речення — про молоко та какао! Читай сміливо!",
            },
        ],
        "done": ("Вітаю, чемпіоне! Ти відкрив уже десять замочків і скоро сам читатимеш казки!", "kapi_level_k_done.wav"),
    },
}