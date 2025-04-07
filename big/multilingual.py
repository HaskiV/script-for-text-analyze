import re
import string
import unicodedata
from langdetect import detect, LangDetectException
import pymorphy2
import nltk
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import spacy
import logging

# Настраиваем логирование
logger = logging.getLogger(__name__)

class MultilingualProcessor:
    """
    Класс для обработки текста на нескольких языках, с фокусом на русский и английский.
    """
    
    def __init__(self):
        """Инициализация процессора с необходимыми языковыми ресурсами."""
        # Загружаем необходимые ресурсы NLTK
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
        except Exception as e:
            logger.error(f"Не удалось загрузить ресурсы NLTK: {e}")
        
        # Инициализируем языковые инструменты
        self.morph_ru = pymorphy2.MorphAnalyzer()
        self.stemmer_ru = SnowballStemmer("russian")
        self.stemmer_en = SnowballStemmer("english")
        
        # Загружаем стоп-слова
        self.stopwords_ru = set(stopwords.words('russian'))
        self.stopwords_en = set(stopwords.words('english'))
        
        # Добавляем пунктуацию в стоп-слова
        self.punctuation = set(string.punctuation + '«»—–')
        self.stopwords_ru.update(self.punctuation)
        self.stopwords_en.update(self.punctuation)
        
        # Пытаемся загрузить модели spaCy
        try:
            self.nlp_ru = spacy.load("ru_core_news_md")
            self.nlp_en = spacy.load("en_core_web_md")
            logger.info("Модели spaCy успешно загружены")
        except Exception as e:
            logger.error(f"Не удалось загрузить модели spaCy: {e}")
            self.nlp_ru = None
            self.nlp_en = None
    
    def detect_language(self, text):
        """
        Определение языка текста.
        
        Args:
            text (str): Входной текст
            
        Returns:
            str: Код языка ('en', 'ru', или 'unknown')
        """
        if not text or not isinstance(text, str):
            return 'unknown'
            
        # Проверяем, содержит ли текст в основном кириллические символы
        cyrillic_pattern = re.compile('[а-яА-Я]')
        latin_pattern = re.compile('[a-zA-Z]')
        
        cyrillic_count = len(cyrillic_pattern.findall(text))
        latin_count = len(latin_pattern.findall(text))
        
        # Быстрая эвристика для коротких текстов
        if cyrillic_count > latin_count:
            return 'ru'
        elif latin_count > cyrillic_count:
            return 'en'
        
        # Для более неоднозначных случаев используем langdetect
        try:
            lang = detect(text)
            if lang == 'ru':
                return 'ru'
            elif lang.startswith('en'):
                return 'en'
            else:
                return lang
        except LangDetectException:
            return 'unknown'
    
    def normalize_text(self, text, language=None):
        """
        Нормализация текста путем удаления лишних пробелов, управляющих символов и т.д.
        
        Args:
            text (str): Входной текст
            language (str, optional): Код языка
            
        Returns:
            str: Нормализованный текст
        """
        if not text:
            return ""
            
        # Определяем язык, если он не предоставлен
        if language is None:
            language = self.detect_language(text)
            
        # Нормализуем Unicode-символы
        text = unicodedata.normalize('NFKC', text)
        
        # Удаляем управляющие символы, кроме переносов строк и табуляции
        text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C' 
                      or ch in ('\n', '\t'))
        
        # Заменяем множественные пробелы на один
        text = re.sub(r'\s+', ' ', text)
        
        # Языковые специфические нормализации
        if language == 'ru':
            # Конвертируем 'ё' в 'е' (обычно в русской обработке текста)
            text = text.replace('ё', 'е').replace('Ё', 'Е')
            
        return text.strip()
    
    def tokenize_sentences(self, text, language=None):
        """
        Разделение текста на предложения.
        
        Args:
            text (str): Входной текст
            language (str, optional): Код языка
            
        Returns:
            list: Список предложений
        """
        if not text:
            return []
            
        # Определяем язык, если он не предоставлен
        if language is None:
            language = self.detect_language(text)
            
        # Сначала нормализуем текст
        text = self.normalize_text(text, language)
        
        # Используем spaCy для более точной сегментации предложений, если доступно
        if (language == 'ru' and self.nlp_ru) or (language == 'en' and self.nlp_en):
            nlp = self.nlp_ru if language == 'ru' else self.nlp_en
            doc = nlp(text)
            return [sent.text for sent in doc.sents]
        
        # Запасной вариант - токенизатор предложений NLTK
        return sent_tokenize(text)
    
    def tokenize_words(self, text, language=None, remove_stopwords=True, 
                      min_length=2, lowercase=True):
        """
        Токенизация текста на слова.
        
        Args:
            text (str): Входной текст
            language (str, optional): Код языка
            remove_stopwords (bool): Удалять ли стоп-слова
            min_length (int): Минимальная длина слова
            lowercase (bool): Приводить ли к нижнему регистру
            
        Returns:
            list: Список токенов
        """
        if not text:
            return []
            
        # Определяем язык, если он не предоставлен
        if language is None:
            language = self.detect_language(text)
            
        # Нормализуем и приводим к нижнему регистру, если запрошено
        text = self.normalize_text(text, language)
        if lowercase:
            text = text.lower()
            
        # Используем соответствующий список стоп-слов
        stops = self.stopwords_ru if language == 'ru' else self.stopwords_en
        
        # Используем spaCy для более точной токенизации, если доступно
        if (language == 'ru' and self.nlp_ru) or (language == 'en' and self.nlp_en):
            nlp = self.nlp_ru if language == 'ru' else self.nlp_en
            doc = nlp(text)
            tokens = [token.text for token in doc 
                     if not (remove_stopwords and token.text.lower() in stops)
                     and len(token.text) >= min_length
                     and not token.is_punct
                     and not token.is_space]
            return tokens
        
        # Запасной вариант - токенизатор слов NLTK
        tokens = word_tokenize(text)
        
        if remove_stopwords:
            tokens = [token for token in tokens 
                     if token.lower() not in stops
                     and len(token) >= min_length
                     and not all(c in string.punctuation for c in token)]
        
        return tokens
    
    def lemmatize(self, word, language=None):
        """
        Лемматизация слова в зависимости от его языка.
        
        Args:
            word (str): Входное слово
            language (str, optional): Код языка
            
        Returns:
            str: Лемматизированное слово
        """
        if not word:
            return ""
            
        # Определяем язык, если он не предоставлен
        if language is None:
            # Для отдельных слов определение языка менее надежно
            # Пытаемся угадать на основе набора символов
            cyrillic_pattern = re.compile('[а-яА-Я]')
            if cyrillic_pattern.search(word):
                language = 'ru'
            else:
                language = 'en'
                
        if language == 'ru':
            # Используем pymorphy2 для русской лемматизации
            return self.morph_ru.parse(word)[0].normal_form
        else:
            # Используем лемматизатор WordNet NLTK для английского
            from nltk.stem import WordNetLemmatizer
            lemmatizer = WordNetLemmatizer()
            return lemmatizer.lemmatize(word)
    
    def extract_keywords(self, text, language=None, top_n=10):
        """
        Извлечение ключевых слов из текста с использованием spaCy.
        
        Args:
            text (str): Входной текст
            language (str, optional): Код языка
            top_n (int): Количество ключевых слов для извлечения
            
        Returns:
            list: Список ключевых слов
        """
        if not text:
            return []
            
        # Определяем язык, если он не предоставлен
        if language is None:
            language = self.detect_language(text)
            
        # Убеждаемся, что у нас есть соответствующая модель spaCy
        if (language == 'ru' and self.nlp_ru is None) or (language == 'en' and self.nlp_en is None):
            logger.warning(f"Нет доступной модели spaCy для {language}. Используем простое извлечение.")
            # Запасной вариант - простая частота слов
            tokens = self.tokenize_words(text, language, remove_stopwords=True)
            from collections import Counter
            return [word for word, _ in Counter(tokens).most_common(top_n)]
            
        # Используем spaCy для извлечения ключевых слов
        nlp = self.nlp_ru if language == 'ru' else self.nlp_en
        doc = nlp(self.normalize_text(text, language))
        
        # Извлекаем существительные и прилагательные как ключевые слова
        keywords = []
        for token in doc:
            if token.pos_ in ['NOUN', 'ADJ'] and not token.is_stop and len(token.text) > 2:
                keywords.append(token.text.lower())
                
        # Подсчитываем частоты и возвращаем топ N
        from collections import Counter
        return [word for word, _ in Counter(keywords).most_common(top_n)]
    
    def extract_named_entities(self, text, language=None):
        """
        Извлечение именованных сущностей из текста с использованием spaCy.
        
        Args:
            text (str): Входной текст
            language (str, optional): Код языка
            
        Returns:
            list: Список кортежей (текст_сущности, тип_сущности)
        """
        if not text:
            return []
            
        # Определяем язык, если он не предоставлен
        if language is None:
            language = self.detect_language(text)
            
        # Убеждаемся, что у нас есть соответствующая модель spaCy
        if (language == 'ru' and self.nlp_ru is None) or (language == 'en' and self.nlp_en is None):
            logger.warning(f"Нет доступной модели spaCy для {language}. Невозможно извлечь именованные сущности.")
            return []
            
        # Используем spaCy для распознавания именованных сущностей
        nlp = self.nlp_ru if language == 'ru' else self.nlp_en
        doc = nlp(self.normalize_text(text, language))
        
        # Извлекаем сущности
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities
    
    def calculate_similarity(self, text1, text2, language=None):
        """
        Расчет семантической схожести между двумя текстами с использованием spaCy.
        
        Args:
            text1 (str): Первый текст
            text2 (str): Второй текст
            language (str, optional): Код языка
            
        Returns:
            float: Оценка схожести от 0 до 1
        """
        if not text1 or not text2:
            return 0.0
            
        # Если тексты идентичны, возвращаем 1.0
        if text1 == text2:
            return 1.0
            
        # Определяем язык, если он не предоставлен (используем первый текст как эталон)
        if language is None:
            language = self.detect_language(text1)
            
        # Убеждаемся, что у нас есть соответствующая модель spaCy
        if (language == 'ru' and self.nlp_ru is None) or (language == 'en' and self.nlp_en is None):
            logger.warning(f"Нет доступной модели spaCy для {language}. Используем перекрытие токенов.")
            
            # Запасной вариант - мера перекрытия токенов
            tokens1 = set(self.tokenize_words(text1, language))
            tokens2 = set(self.tokenize_words(text2, language))
            
            if not tokens1 or not tokens2:
                return 0.0
                
            # Схожесть Жаккара: пересечение над объединением
            intersection = len(tokens1.intersection(tokens2))
            union = len(tokens1.union(tokens2))
            
            return intersection / union if union > 0 else 0.0
            
        # Используем spaCy для семантической схожести
        nlp = self.nlp_ru if language == 'ru' else self.nlp_en
        doc1 = nlp(self.normalize_text(text1, language))
        doc2 = nlp(self.normalize_text(text2, language))
        
        return doc1.similarity(doc2)
    
    def get_language_name(self, language_code):
        """
        Получение человекочитаемого названия языка.
        
        Args:
            language_code (str): Код языка
            
        Returns:
            str: Название языка
        """
        language_names = {
            'ru': 'Русский',
            'en': 'Английский',
            'unknown': 'Неизвестный'
        }
        return language_names.get(language_code, language_code)
    
    def process_bilingual_text(self, text):
        """
        Обработка текста, который может содержать несколько языков.
        
        Args:
            text (str): Входной текст
            
        Returns:
            dict: Словарь с результатами обработки для каждого языка
        """
        if not text:
            return {}
            
        # Разбиваем текст на предложения
        sentences = self.tokenize_sentences(text)
        
        # Группируем предложения по языку
        sentences_by_lang = {}
        for sentence in sentences:
            lang = self.detect_language(sentence)
            if lang not in sentences_by_lang:
                sentences_by_lang[lang] = []
            sentences_by_lang[lang].append(sentence)
        
        # Обрабатываем каждый язык отдельно
        results = {}
        for lang, lang_sentences in sentences_by_lang.items():
            # Пропускаем неизвестный язык
            if lang == 'unknown':
                continue
                
            # Объединяем предложения
            lang_text = ' '.join(lang_sentences)
            
            # Извлекаем ключевые слова
            keywords = self.extract_keywords(lang_text, language=lang)
            
            # Извлекаем именованные сущности
            entities = self.extract_named_entities(lang_text, language=lang)
            
            # Добавляем в результаты
            results[lang] = {
                'language': self.get_language_name(lang),
                'text': lang_text,
                'sentences_count': len(lang_sentences),
                'keywords': keywords,
                'entities': entities
            }
        
        return results
