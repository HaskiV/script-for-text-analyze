import os
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
import seaborn as sns
import pandas as pd
import logging
from typing import Dict, List, Tuple, Union, Optional
import json
from text_analyzer import TextAnalyzer
from collections import Counter

logger = logging.getLogger(__name__)

class ResultVisualizer:
    """
    Класс для визуализации результатов анализа текста.
    """
    
    def __init__(self, output_dir: str = "results"):
        """
        Инициализация визуализатора.
        
        Args:
            output_dir (str): Директория для сохранения визуализаций
        """
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Создана директория для результатов: {output_dir}")
    
    def create_word_cloud(self, word_frequencies: Dict[str, int], 
                         title: str = "Облако слов", 
                         filename: Optional[str] = None) -> str:
        """
        Создание облака слов из частот слов.
        
        Args:
            word_frequencies (dict): Словарь частот слов
            title (str): Заголовок визуализации
            filename (str, optional): Имя файла для сохранения
            
        Returns:
            str: Путь к сохраненному файлу
        """
        if not word_frequencies:
            logger.warning("Нет данных для создания облака слов")
            return None
            
        # Создаем облако слов
        wordcloud = WordCloud(width=800, height=400, 
                            background_color='white',
                            max_words=100).generate_from_frequencies(word_frequencies)
        
        # Создаем фигуру
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        
        # Сохраняем изображение
        if filename is None:
            filename = "wordcloud.png"
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info(f"Облако слов сохранено в: {output_path}")
        return output_path
    
    def create_bar_chart(self, data: List[Tuple[str, int]], 
                        title: str = "Частоты слов",
                        xlabel: str = "Слово",
                        ylabel: str = "Частота",
                        filename: Optional[str] = None) -> str:
        """
        Создание столбчатой диаграммы.
        
        Args:
            data (list): Список кортежей (метка, значение)
            title (str): Заголовок диаграммы
            xlabel (str): Подпись оси X
            ylabel (str): Подпись оси Y
            filename (str, optional): Имя файла для сохранения
            
        Returns:
            str: Путь к сохраненному файлу
        """
        if not data:
            logger.warning("Нет данных для создания столбчатой диаграммы")
            return None
            
        # Создаем DataFrame
        df = pd.DataFrame(data, columns=[xlabel, ylabel])
        
        # Создаем диаграмму
        plt.figure(figsize=(12, 6))
        sns.barplot(x=xlabel, y=ylabel, data=df)
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Сохраняем изображение
        if filename is None:
            filename = "bar_chart.png"
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info(f"Столбчатая диаграмма сохранена в: {output_path}")
        return output_path
    
    def create_heatmap(self, data: Dict[str, List[float]], 
                      title: str = "Тепловая карта схожести",
                      filename: Optional[str] = None) -> str:
        """
        Создание тепловой карты.
        
        Args:
            data (dict): Данные для тепловой карты в формате {метка: [значения]}
            title (str): Заголовок карты
            filename (str, optional): Имя файла для сохранения
            
        Returns:
            str: Путь к сохраненному файлу
        """
        if not data:
            logger.warning("Нет данных для создания тепловой карты")
            return ""
            
        logger.info(f"Создание тепловой карты с {len(data)} категориями")
        
        # Создаем DataFrame
        df = pd.DataFrame(data)
        
        # Создаем тепловую карту
        plt.figure(figsize=(10, 8))
        sns.heatmap(df, annot=True, cmap='YlOrRd', fmt='.2f')
        plt.title(title)
        plt.tight_layout()
        
        # Сохраняем изображение
        if filename is None:
            filename = "heatmap.png"
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info(f"Тепловая карта сохранена в: {output_path}")
        return output_path
    
    def create_intent_distribution_chart(self, extracted_intents: Dict[str, List[Dict]]) -> str:
        """
        Создание диаграммы распределения интентов.
        
        Args:
            extracted_intents (dict): Словарь извлеченных интентов в формате:
                {
                    'intent_name': [
                        {'text': 'текст', 'similarity': 0.85},
                        ...
                    ]
                }
            
        Returns:
            str: Путь к сохраненному файлу
        """
        if not extracted_intents:
            logger.warning("Нет данных для создания диаграммы распределения интентов")
            return ""
            
        logger.info(f"Создание диаграммы распределения для {len(extracted_intents)} интентов")
        
        # Подсчитываем количество и среднюю схожесть для каждого интента
        intent_stats = {}
        for intent_name, matches in extracted_intents.items():
            if matches:
                intent_stats[intent_name] = {
                    'count': len(matches),
                    'avg_similarity': np.mean([m.get('similarity', 0.0) for m in matches])
                }
        
        if not intent_stats:
            logger.warning("Нет совпадений для создания диаграммы распределения")
            return ""
            
        # Создаем DataFrame для визуализации
        df = pd.DataFrame(intent_stats).T
        df = df.sort_values('count', ascending=False)
        
        # Создаем фигуру с двумя подграфиками
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Первый график - количество совпадений
        bars = ax1.bar(df.index, df['count'])
        ax1.set_title('Количество совпадений по интентам')
        ax1.set_xlabel('Интент')
        ax1.set_ylabel('Количество')
        ax1.tick_params(axis='x', rotation=45)
        
        # Добавляем значения на столбцы
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        # Второй график - средняя схожесть
        bars = ax2.bar(df.index, df['avg_similarity'])
        ax2.set_title('Средняя схожесть по интентам')
        ax2.set_xlabel('Интент')
        ax2.set_ylabel('Средняя схожесть')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(0, 1)  # Ограничиваем шкалу от 0 до 1
        
        # Добавляем значения на столбцы
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 'intent_distribution.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info(f"Диаграмма распределения интентов сохранена в: {output_path}")
        return output_path
    
    def create_intent_similarity_heatmap(self, extracted_intents: Dict[str, List[Dict]]) -> str:
        """
        Создание тепловой карты схожести намерений.
        
        Args:
            extracted_intents (dict): Словарь извлеченных интентов в формате:
                {
                    'intent_name': [
                        {'text': 'текст', 'similarity': 0.85},
                        ...
                    ]
                }
            
        Returns:
            str: Путь к сохраненному файлу
        """
        if not extracted_intents:
            logger.warning("Нет данных для создания тепловой карты схожести намерений")
            return ""
            
        logger.info(f"Создание тепловой карты схожести для {len(extracted_intents)} интентов")
        
        # Создаем матрицу схожести между текстами разных интентов
        intents = list(extracted_intents.keys())
        similarity_matrix = np.zeros((len(intents), len(intents)))
        
        # Для каждой пары интентов вычисляем среднюю схожесть между их текстами
        for i, intent1 in enumerate(intents):
            for j, intent2 in enumerate(intents):
                if i == j:
                    # На диагонали ставим 1.0 (полное совпадение)
                    similarity_matrix[i][j] = 1.0
                else:
                    # Вычисляем среднюю схожесть между текстами двух интентов
                    similarities = []
                    for match1 in extracted_intents[intent1]:
                        for match2 in extracted_intents[intent2]:
                            # Используем минимальное значение схожести из пары
                            similarities.append(min(match1.get('similarity', 0.0), 
                                                 match2.get('similarity', 0.0)))
                    
                    similarity_matrix[i][j] = np.mean(similarities) if similarities else 0.0
        
        # Создаем DataFrame для тепловой карты
        df = pd.DataFrame(similarity_matrix, index=intents, columns=intents)
        
        # Создаем тепловую карту
        plt.figure(figsize=(10, 8))
        sns.heatmap(df, annot=True, cmap='YlOrRd', fmt='.2f')
        plt.title("Схожесть намерений")
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 'intent_similarity_heatmap.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info(f"Тепловая карта схожести намерений сохранена в: {output_path}")
        return output_path
    
    def create_html_report(self, results: Dict, analyzer: Optional[TextAnalyzer] = None, title: str = "Отчет по анализу текста") -> str:
        """
        Создание HTML-отчета с визуализациями.
        
        Args:
            results (dict): Результаты анализа
            analyzer (TextAnalyzer, optional): Анализатор текста
            title (str): Заголовок отчета
            
        Returns:
            str: Путь к сохраненному файлу
        """
        # Создаем HTML-отчет
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                .visualization {{ margin: 20px 0; }}
                img {{ max-width: 100%; height: auto; }}
                .intent-section {{ margin: 20px 0; padding: 15px; background-color: #f5f5f5; border-radius: 5px; }}
                .intent-header {{ font-size: 1.2em; font-weight: bold; margin-bottom: 10px; }}
                .match-item {{ margin: 5px 0; padding: 5px; background-color: white; border-radius: 3px; }}
                .similarity {{ color: #666; font-size: 0.9em; }}
                .example {{ color: #888; font-style: italic; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
        """
        
        # Добавляем облако слов
        if 'word_frequencies' in results:
            wordcloud_path = self.create_word_cloud(dict(results['word_frequencies'][:100]))
            if wordcloud_path:
                html_content += f"""
                <div class="visualization">
                    <h2>Облако слов</h2>
                    <img src="{os.path.relpath(wordcloud_path, self.output_dir)}" alt="Облако слов">
                </div>
                """
        
        # Добавляем столбчатую диаграмму
        if 'word_frequencies' in results:
            bar_chart_path = self.create_bar_chart(results['word_frequencies'][:20])
            if bar_chart_path:
                html_content += f"""
                <div class="visualization">
                    <h2>Топ-20 частых слов</h2>
                    <img src="{os.path.relpath(bar_chart_path, self.output_dir)}" alt="Столбчатая диаграмма">
                </div>
                """
        
        # Добавляем тепловую карту схожести намерений
        if 'extracted_intents' in results and results['extracted_intents']:
            heatmap_path = self.create_intent_similarity_heatmap(results['extracted_intents'])
            if heatmap_path:
                html_content += f"""
                <div class="visualization">
                    <h2>Схожесть намерений</h2>
                    <img src="{os.path.relpath(heatmap_path, self.output_dir)}" alt="Тепловая карта">
                </div>
                """
        
        # Добавляем диаграмму распределения намерений
        if 'extracted_intents' in results and results['extracted_intents']:
            distribution_path = self.create_intent_distribution_chart(results['extracted_intents'])
            if distribution_path:
                html_content += f"""
                <div class="visualization">
                    <h2>Распределение намерений</h2>
                    <img src="{os.path.relpath(distribution_path, self.output_dir)}" alt="Распределение намерений">
                </div>
                """
        
        # Добавляем список похожих фраз для каждого интента
        if 'extracted_intents' in results and results['extracted_intents']:
            html_content += """
                <div class="visualization">
                    <h2>Похожие фразы по интентам</h2>
            """
            
            for intent_name, matches in results['extracted_intents'].items():
                if matches:
                    html_content += f"""
                        <div class="intent-section">
                            <div class="intent-header">{intent_name}</div>
                    """
                    
                    # Добавляем примеры интента, если они есть
                    if analyzer and hasattr(analyzer, 'intent_examples') and intent_name in analyzer.intent_examples:
                        examples = analyzer.intent_examples[intent_name]
                        html_content += """
                            <div class="example">
                                <strong>Примеры интента:</strong><br>
                        """
                        for example in examples:
                            html_content += f"• {example}<br>"
                        html_content += """
                            </div>
                        """
                    
                    # Добавляем найденные совпадения
                    for match in matches:
                        text = match.get('text', '')
                        similarity = match.get('similarity', 0.0)
                        html_content += f"""
                            <div class="match-item">
                                <div>{text}</div>
                                <div class="similarity">Схожесть: {similarity:.2f}</div>
                            </div>
                        """
                    
                    html_content += """
                        </div>
                    """
            
            html_content += """
                </div>
            """
        
        # Завершаем HTML
        html_content += """
        </body>
        </html>
        """
        
        # Сохраняем HTML-отчет
        output_path = os.path.join(self.output_dir, "report.html")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML-отчет сохранен в: {output_path}")
        return output_path
