import os
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from tqdm import tqdm
import chardet
import openpyxl
import csv
from concurrent.futures import ProcessPoolExecutor
import spacy
from typing import List, Dict, Tuple, Union, Optional
from multilingual import MultilingualProcessor

# Настраиваем логирование
logger = logging.getLogger(__name__)

# Download required NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    
    # Добавляем загрузку punkt_tab
    try:
        nltk.download('punkt_tab', quiet=True)
    except:
        # Если ресурс не найден в репозитории, используем альтернативный подход
        nltk.download('punkt', quiet=True)
        
    logger.info("NLTK resources downloaded successfully")
except Exception as e:
    logger.error(f"Не удалось загрузить ресурсы NLTK: {e}")

# Load spaCy models for English and Russian
try:
    nlp_en = spacy.load("en_core_web_md")
    nlp_ru = spacy.load("ru_core_news_md")
    
    # Увеличиваем максимальную длину текста
    nlp_en.max_length = 3000000  # Увеличиваем еще больше
    nlp_ru.max_length = 3000000
    
    logger.info("spaCy models loaded successfully")
except Exception as e:
    logger.error(f"Не удалось загрузить spaCy модели: {e}")
    logger.info("Устанавливаем spaCy модели...")
    # Attempt to install models
    import subprocess
    subprocess.call("python -m spacy download en_core_web_md", shell=True)
    subprocess.call("python -m spacy download ru_core_news_md", shell=True)
    try:
        nlp_en = spacy.load("en_core_web_md")
        nlp_ru = spacy.load("ru_core_news_md")
    except Exception as e:
        logger.error(f"Не удалось загрузить spaCy модели: {e}")
        logger.info("Откатываемся к маленьким моделям...")
        subprocess.call("python -m spacy download en_core_web_sm", shell=True)
        subprocess.call("python -m spacy download ru_core_news_sm", shell=True)
        nlp_en = spacy.load("en_core_web_sm")
        nlp_ru = spacy.load("ru_core_news_sm")

class TextAnalyzer:
    """
    Класс для анализа текста, включая извлечение намерений и частот слов.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Инициализация анализатора текста.
        
        Args:
            config (dict, optional): Конфигурация анализатора
        """
        self.config = config or {}
        self.multilingual_processor = MultilingualProcessor()
        self.vectorizer = TfidfVectorizer()
        self.intent_examples = self.config.get('intents', {})
        self.similarity_threshold = self.config.get('similarity_threshold', 0.6)
        
        # Загружаем необходимые ресурсы NLTK
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
        except Exception as e:
            logger.error(f"Не удалось загрузить ресурсы NLTK: {e}")
    
    def set_intent_examples(self, intent_examples: Dict[str, List[str]]) -> None:
        """
        Установка примеров намерений.
        
        Args:
            intent_examples (dict): Словарь примеров намерений в формате {имя_намерения: [пример1, пример2, ...]}
        """
        self.intent_examples = intent_examples
        logger.info(f"Установлены примеры намерений: {len(intent_examples)} типов")
    
    def process_text(self, text: str, language: Optional[str] = None) -> Dict:
        """
        Обработка текста для извлечения намерений и частот слов.
        
        Args:
            text (str): Входной текст
            language (str, optional): Код языка
            
        Returns:
            dict: Результаты анализа
        """
        if not text:
            return {'intents': [], 'word_frequencies': []}
            
        # Определяем язык, если он не предоставлен
        if language is None:
            language = self.multilingual_processor.detect_language(text)
            
        # Нормализуем текст
        normalized_text = self.multilingual_processor.normalize_text(text, language)
        
        # Извлекаем намерения
        intents = self.extract_intents(normalized_text, language)
        
        # Извлекаем частоты слов
        word_frequencies = self.extract_word_frequencies(normalized_text, language)
        
        return {
            'intents': intents,
            'word_frequencies': word_frequencies
        }
    
    def extract_intents(self, text: str, language: Optional[str] = None) -> List[Dict]:
        """
        Извлечение намерений из текста.
        
        Args:
            text (str): Входной текст
            language (str, optional): Код языка
            
        Returns:
            list: Список найденных намерений
        """
        if not text or not self.intent_examples:
            return []
            
        # Определяем язык, если он не предоставлен
        if language is None:
            language = self.multilingual_processor.detect_language(text)
            
        # Токенизируем текст
        tokens = self.multilingual_processor.tokenize_words(text, language)
        if not tokens:
            return []
            
        # Создаем TF-IDF векторы для текста и примеров намерений
        all_texts = [text] + [example for examples in self.intent_examples.values() 
                            for example in examples]
        try:
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        except Exception as e:
            logger.error(f"Ошибка при создании TF-IDF векторов: {e}")
            return []
            
        # Рассчитываем схожесть с каждым примером намерения
        text_vector = tfidf_matrix[0]
        intent_vectors = tfidf_matrix[1:]
        
        similarities = cosine_similarity(text_vector, intent_vectors)[0]
        
        # Группируем схожести по намерениям
        intent_similarities = {}
        current_idx = 0
        for intent, examples in self.intent_examples.items():
            intent_similarities[intent] = similarities[current_idx:current_idx + len(examples)]
            current_idx += len(examples)
            
        # Находим максимальную схожесть для каждого намерения
        results = []
        for intent, similarities in intent_similarities.items():
            max_similarity = max(similarities)
            if max_similarity >= self.similarity_threshold:
                results.append({
                    'intent': intent,
                    'confidence': float(max_similarity)
                })
                
        return sorted(results, key=lambda x: x['confidence'], reverse=True)
    
    def extract_word_frequencies(self, text: str, language: Optional[str] = None) -> List[Tuple[str, int]]:
        """
        Извлечение частот слов из текста.
        
        Args:
            text (str): Входной текст
            language (str, optional): Код языка
            
        Returns:
            list: Список кортежей (слово, частота)
        """
        if not text:
            return []
            
        # Определяем язык, если он не предоставлен
        if language is None:
            language = self.multilingual_processor.detect_language(text)
            
        # Токенизируем текст
        tokens = self.multilingual_processor.tokenize_words(text, language)
        if not tokens:
            return []
            
        # Подсчитываем частоты
        frequencies = Counter(tokens)
        
        # Сортируем по частоте
        return sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
    
    def analyze_file(self, file_path: str, language: Optional[str] = None) -> Dict:
        """
        Анализ содержимого файла.
        
        Args:
            file_path (str): Путь к файлу
            language (str, optional): Код языка
            
        Returns:
            dict: Результаты анализа
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            logger.error(f"Не удалось прочитать файл {file_path}: {e}")
            return {'intents': [], 'word_frequencies': []}
            
        return self.process_text(text, language)
    
    def analyze_directory(self, directory_path: str, language: Optional[str] = None) -> Dict:
        """
        Анализ всех текстовых файлов в директории.
        
        Args:
            directory_path (str): Путь к директории
            language (str, optional): Код языка
            
        Returns:
            dict: Агрегированные результаты анализа
        """
        import os
        
        all_intents = []
        all_word_frequencies = Counter()
        
        for filename in os.listdir(directory_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(directory_path, filename)
                results = self.analyze_file(file_path, language)
                
                all_intents.extend(results['intents'])
                all_word_frequencies.update(dict(results['word_frequencies']))
                
        return {
            'intents': sorted(all_intents, key=lambda x: x['confidence'], reverse=True),
            'word_frequencies': sorted(all_word_frequencies.items(), 
                                    key=lambda x: x[1], reverse=True)
        }


def main():
    """
    Main function to demonstrate the TextAnalyzer.
    """
    # Create analyzer with default settings
    analyzer = TextAnalyzer(chunk_size=1000, language='auto', num_workers=4)
    
    # Add intent examples (in both English and Russian)
    analyzer.add_intent_examples("complaint", [
        "I am not satisfied with your service",
        "This product doesn't work as advertised",
        "I want to file a complaint about the quality",
        "Я недоволен вашим обслуживанием",
        "Этот продукт не работает, как рекламировалось",
        "Я хочу подать жалобу на качество"
    ])
    
    analyzer.add_intent_examples("inquiry", [
        "How can I use this feature?",
        "What is the price of this product?",
        "I would like to know more about your services",
        "Как я могу использовать эту функцию?",
        "Сколько стоит этот продукт?",
        "Я хотел бы узнать больше о ваших услугах"
    ])
    
    # Define file paths
    file_paths = [
        "example.txt",
        "data.csv",
        "report.xlsx"
    ]
    
    # Process files
    results = analyzer.process_files(
        file_paths,
        calculate_frequencies=True,
        extract_intents=True,
        similarity_threshold=0.7,
        top_n_words=100
    )
    
    # Save results
    analyzer.save_results(results)
    
    # Print some results
    if 'word_frequencies' in results:
        print("\nTop 10 Most Frequent Words:")
        for word, freq in results['word_frequencies'][:10]:
            print(f"{word}: {freq}")
    
    if 'extracted_intents' in results:
        print("\nExtracted Intents:")
        for intent, matches in results['extracted_intents'].items():
            print(f"\n{intent.upper()} - {len(matches)} matches found")
            for match in matches[:5]:  # Print first 5 matches
                print(f"- {match['text']} (similarity: {match['similarity']:.2f})")


if __name__ == "__main__":
    main()
