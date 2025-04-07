#!/usr/bin/env python3
"""
Многоязычный инструмент анализа текста

Этот скрипт обрабатывает текстовые данные из различных форматов файлов,
идентифицирует частые слова и извлекает намерения на основе примеров.
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import time

# Импортируем наши модули
from text_analyzer import TextAnalyzer
from config_loader import ConfigLoader
from visualization import ResultVisualizer
from multilingual import MultilingualProcessor

# Удаляем неправильный импорт
# from text_processing_module import process_text_chunk

# Настраиваем логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("text_analysis.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def parse_arguments():
    """Разбор аргументов командной строки."""
    parser = argparse.ArgumentParser(description="Многоязычный инструмент анализа текста")
    
    parser.add_argument("--config", "-c", type=str, default="config.yml",
                        help="Путь к файлу конфигурации (YAML или JSON)")
    
    parser.add_argument("--files", "-f", type=str, nargs="+",
                        help="Файлы для анализа (переопределяет список файлов из конфига)")
    
    parser.add_argument("--output-dir", "-o", type=str,
                        help="Директория для результатов (переопределяет конфиг)")
    
    parser.add_argument("--create-config", action="store_true",
                        help="Создать файл конфигурации по умолчанию и выйти")
    
    parser.add_argument("--chunk-size", type=int,
                        help="Размер чанков текста для обработки (переопределяет конфиг)")
    
    parser.add_argument("--workers", "-w", type=int,
                        help="Количество рабочих процессов (переопределяет конфиг)")
    
    parser.add_argument("--language", "-l", type=str, choices=["auto", "en", "ru"],
                        help="Принудительная обработка на определенном языке (переопределяет конфиг)")
    
    return parser.parse_args()

def load_file_content(file_path):
    """
    Загрузка содержимого файла в зависимости от его расширения.
    
    Args:
        file_path (str): Путь к файлу
        
    Returns:
        list: Список текстовых чанков из файла
    """
    # Преобразуем путь к стандартному формату для OS
    file_path = os.path.normpath(file_path)
    # Получаем абсолютный путь
    abs_path = os.path.abspath(file_path)
    
    logger.info(f"Попытка доступа к файлу: {abs_path}")
    
    if not os.path.exists(abs_path):
        logger.error(f"Файл не найден: {file_path} (абсолютный путь: {abs_path})")
        return []
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return [f.read()]
                
        elif file_ext == '.csv':
            import pandas as pd
            df = pd.read_csv(file_path)
            # Объединяем все текстовые колонки
            return [' '.join(df[col].astype(str)) for col in df.columns if df[col].dtype == 'object']
            
        elif file_ext == '.xlsx':
            import pandas as pd
            # Читаем все листы
            xlsx = pd.ExcelFile(file_path)
            text_chunks = []
            
            for sheet_name in xlsx.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                # Объединяем все текстовые колонки
                text_chunks.extend([' '.join(df[col].astype(str)) for col in df.columns if df[col].dtype == 'object'])
                
            return text_chunks
            
        else:
            logger.warning(f"Неподдерживаемый формат файла: {file_ext}")
            return []
            
    except Exception as e:
        logger.error(f"Ошибка загрузки файла {file_path}: {str(e)}")
        return []

def process_chunk(chunk, config):
    """
    Обработка одного текстового чанка.
    
    Args:
        chunk (str): Текстовый чанк
        config (dict): Словарь конфигурации
        
    Returns:
        dict: Результаты обработки
    """
    try:
        # Инициализируем процессор
        processor = MultilingualProcessor()
        language = processor.detect_language(chunk)
        
        # Токенизируем текст
        tokens = processor.tokenize_words(chunk, language=language, remove_stopwords=True)
        
        # Подсчитываем частоты слов
        from collections import Counter
        word_frequencies = Counter(tokens)
        
        # Извлекаем намерения, если доступны примеры
        extracted_intents = {}
        if 'intent_examples' in config and config['intent_examples']:
            for intent_name, examples in config['intent_examples'].items():
                extracted_intents[intent_name] = []
                for example in examples:
                    similarity = processor.calculate_similarity(chunk, example, language=language)
                    if similarity >= config.get('similarity_threshold', 0.6):
                        extracted_intents[intent_name].append({
                            'text': chunk[:100] + '...' if len(chunk) > 100 else chunk,
                            'similarity': similarity
                        })
        
        return {
            'word_frequencies': dict(word_frequencies),
            'extracted_intents': extracted_intents,
            'language': language,
            'text_length': len(chunk)
        }
    except Exception as e:
        logger.error(f"Ошибка в process_chunk: {str(e)}")
        return {
            'word_frequencies': {},
            'extracted_intents': {},
            'language': 'unknown',
            'text_length': len(chunk) if chunk else 0
        }

def merge_results(results_list):
    """
    Объединение результатов из нескольких чанков.
    
    Args:
        results_list (list): Список словарей результатов
        
    Returns:
        dict: Объединенные результаты
    """
    logger.info("Объединение результатов из всех чанков")
    
    merged = {
        'word_frequencies': {},
        'extracted_intents': {},
        'languages': {},
        'total_text_length': 0
    }
    
    # Объединяем частоты слов
    for results in results_list:
        # Добавляем частоты слов
        for word, freq in results['word_frequencies'].items():
            if word in merged['word_frequencies']:
                merged['word_frequencies'][word] += freq
            else:
                merged['word_frequencies'][word] = freq
        
        # Добавляем извлеченные намерения
        for intent, matches in results['extracted_intents'].items():
            if intent not in merged['extracted_intents']:
                merged['extracted_intents'][intent] = []
            merged['extracted_intents'][intent].extend(matches)
        
        # Подсчитываем языки
        lang = results['language']
        if lang in merged['languages']:
            merged['languages'][lang] += 1
        else:
            merged['languages'][lang] = 1
        
        # Добавляем длину текста
        merged['total_text_length'] += results['text_length']
    
    # Сортируем частоты слов и конвертируем в список кортежей для визуализации
    merged['word_frequencies'] = sorted(
        merged['word_frequencies'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return merged

def process_text_chunk(chunk, analyzer, settings):
    # Ваша логика обработки здесь
    pass

def main():
    """Основная функция для запуска инструмента анализа текста."""
    start_time = time.time()
    
    # Разбираем аргументы командной строки
    args = parse_arguments()
    
    # Создаем конфиг по умолчанию, если запрошено
    if args.create_config:
        from config_loader import create_default_config
        create_default_config(args.config)
        logger.info(f"Конфигурация по умолчанию создана в: {args.config}")
        return
    
    # Загружаем конфигурацию
    try:
        config = ConfigLoader.load_config(args.config)
        logger.info(f"Конфигурация загружена из {args.config}")
    except Exception as e:
        logger.error(f"Ошибка загрузки конфигурации: {str(e)}")
        return
    
    # Получаем настройки анализа
    settings = ConfigLoader.get_analysis_settings(config)
    
    # Переопределяем настройки аргументами командной строки, если они предоставлены
    if args.output_dir:
        settings['output_dir'] = args.output_dir
    if args.chunk_size:
        settings['chunk_size'] = args.chunk_size
    if args.workers:
        settings['num_workers'] = args.workers
    if args.language:
        settings['language'] = args.language
    
    # Создаем директорию для результатов, если она не существует
    if not os.path.exists(settings['output_dir']):
        os.makedirs(settings['output_dir'])
    
    # Инициализируем анализатор текста
    analyzer = TextAnalyzer()
    
    # Загружаем примеры намерений из конфига
    intent_examples = ConfigLoader.get_intent_examples(config)
    if intent_examples:
        logger.info(f"Загружено {len(intent_examples)} примеров намерений")
        for intent_name, examples in intent_examples.items():
            logger.info(f"Интент '{intent_name}': {len(examples)} примеров")
        analyzer.set_intent_examples(intent_examples)
    else:
        logger.warning("Не найдены примеры намерений в конфигурации")
    
    # Получаем файлы для обработки
    files_to_process = []
    if args.files:
        files_to_process = args.files
    elif 'files' in config:
        files_to_process = config['files']
    
    if not files_to_process:
        logger.error("Не указаны файлы для обработки")
        return
    
    # Загружаем содержимое всех файлов
    all_text_chunks = []
    for file_path in files_to_process:
        chunks = load_file_content(file_path)
        all_text_chunks.extend(chunks)
    
    if not all_text_chunks:
        logger.error("Не удалось загрузить текстовое содержимое из файлов")
        return
    
    logger.info(f"Загружено {len(all_text_chunks)} текстовых чанков из {len(files_to_process)} файлов")
    
    # Разбиваем большие чанки на меньшие на основе chunk_size
    processed_chunks = []
    for chunk in all_text_chunks:
        if len(chunk) > settings['chunk_size']:
            # Разбиваем на меньшие чанки
            words = chunk.split()
            for i in range(0, len(words), settings['chunk_size']):
                end = min(i + settings['chunk_size'], len(words))
                processed_chunks.append(' '.join(words[i:end]))
        else:
            processed_chunks.append(chunk)
    
    logger.info(f"Обработка {len(processed_chunks)} чанков с {settings['num_workers']} рабочими процессами")
    
    # Обрабатываем чанки параллельно
    results_list = []
    with ProcessPoolExecutor(max_workers=settings['num_workers']) as executor:
        # Отправляем задачи с правильной конфигурацией
        futures = [executor.submit(process_chunk, chunk, config) for chunk in processed_chunks]
        
        for future in futures:
            try:
                result = future.result()
                results_list.append(result)
            except Exception as e:
                logger.error(f"Ошибка обработки чанка: {str(e)}")
    
    # Объединяем результаты из всех чанков
    merged_results = merge_results(results_list)
    
    # Инициализируем визуализатор
    visualizer = ResultVisualizer(output_dir=settings['output_dir'])
    
    # Создаем визуализации
    visualizer.create_word_cloud(dict(merged_results['word_frequencies'][:100]))
    visualizer.create_bar_chart(
        merged_results['word_frequencies'][:20],
        "Слово",
        "Частота",
        "Топ-20 самых частых слов"
    )
    
    if merged_results['extracted_intents']:
        visualizer.create_intent_distribution_chart(merged_results['extracted_intents'])
        visualizer.create_intent_similarity_heatmap(merged_results['extracted_intents'])
    
    # Создаем HTML-отчет
    report_path = visualizer.create_html_report(merged_results, analyzer)
    
    # Сохраняем результаты в JSON
    results_path = os.path.join(settings['output_dir'], 'analysis_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        # Конвертируем в сериализуемый формат
        serializable_results = {
            'word_frequencies': dict(merged_results['word_frequencies']),
            'languages': merged_results['languages'],
            'total_text_length': merged_results['total_text_length'],
            # Упрощаем результаты намерений для JSON-сериализации
            'extracted_intents': {
                intent: [
                    {'text': match['text'], 'similarity': match['similarity']}
                    for match in matches
                ]
                for intent, matches in merged_results['extracted_intents'].items()
            }
        }
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Анализ завершен за {elapsed_time:.2f} секунд")
    logger.info(f"Результаты сохранены в {settings['output_dir']}")
    logger.info(f"HTML-отчет доступен по адресу: {report_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Необработанное исключение: {str(e)}")
        sys.exit(1)