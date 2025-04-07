import argparse
import logging
import os
import sys
from typing import List, Optional
from text_analyzer import TextAnalyzer
from visualization import ResultVisualizer

# Настраиваем логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("text_analysis.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class TextAnalysisCLI:
    def __init__(self):
        self.analyzer = None
        self.visualizer = None
        self.settings = {
            'chunk_size': 1000,
            'language': 'auto',
            'workers': 4,
            'threshold': 0.6,
            'top_words': 50,
            'output_dir': 'results'
        }
        self.current_file = None
        self.results = None

    def clear_screen(self):
        """Очистка экрана консоли"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def print_header(self):
        """Вывод заголовка программы"""
        self.clear_screen()
        print("=" * 50)
        print("Инструмент быстрого анализа текста".center(50))
        print("=" * 50)
        print()

    def print_settings(self):
        """Вывод текущих настроек"""
        print("\nТекущие настройки:")
        print("-" * 30)
        for key, value in self.settings.items():
            print(f"{key}: {value}")
        print("-" * 30)

    def change_setting(self):
        """Изменение настроек"""
        self.print_settings()
        print("\nВыберите параметр для изменения:")
        settings_list = list(self.settings.keys())
        for i, setting in enumerate(settings_list, 1):
            print(f"{i}. {setting}")
        
        try:
            choice = int(input("\nВведите номер параметра (0 для отмены): "))
            if choice == 0:
                return
            
            if 1 <= choice <= len(settings_list):
                setting = settings_list[choice - 1]
                new_value = input(f"Введите новое значение для {setting}: ")
                
                # Преобразование типов
                if setting in ['chunk_size', 'workers', 'top_words']:
                    new_value = int(new_value)
                elif setting in ['threshold']:
                    new_value = float(new_value)
                
                self.settings[setting] = new_value
                print(f"Параметр {setting} изменен на {new_value}")
            else:
                print("Неверный выбор")
        except ValueError:
            print("Ошибка: введите число")

    def select_file(self):
        """Выбор файла для анализа"""
        while True:
            self.print_header()
            print("Выберите способ выбора файла:")
            print("1. Выбрать файл из текущей директории")
            print("2. Указать полный путь к файлу")
            print("0. Вернуться в главное меню")
            
            try:
                choice = int(input("\nВыберите действие: "))
                
                if choice == 0:
                    return
                elif choice == 1:
                    self.select_file_from_current_dir()
                    break
                elif choice == 2:
                    self.select_file_by_path()
                    break
                else:
                    print("Неверный выбор")
                    input("\nНажмите Enter для продолжения...")
                    
            except ValueError:
                print("Ошибка: введите число")
                input("\nНажмите Enter для продолжения...")

    def select_file_from_current_dir(self):
        """Выбор файла из текущей директории"""
        print("\nВыберите файл для анализа:")
        files = [f for f in os.listdir('.') if os.path.isfile(f)]
        if not files:
            print("В текущей директории нет файлов")
            return
            
        for i, file in enumerate(files, 1):
            print(f"{i}. {file}")
        
        try:
            choice = int(input("\nВведите номер файла (0 для отмены): "))
            if choice == 0:
                return
            
            if 1 <= choice <= len(files):
                self.current_file = files[choice - 1]
                print(f"Выбран файл: {self.current_file}")
            else:
                print("Неверный выбор")
        except ValueError:
            print("Ошибка: введите число")

    def select_file_by_path(self):
        """Выбор файла по полному пути"""
        while True:
            print("\nВведите полный путь к файлу (или 0 для отмены):")
            path = input().strip()
            
            if path == '0':
                return
                
            if not os.path.exists(path):
                print("Файл не найден. Проверьте путь и попробуйте снова.")
                continue
                
            if not os.path.isfile(path):
                print("Указанный путь не является файлом.")
                continue
                
            self.current_file = path
            print(f"Выбран файл: {self.current_file}")
            break

    def run_analysis(self):
        """Запуск анализа"""
        if not self.current_file:
            print("Сначала выберите файл для анализа")
            return

        try:
            # Создаем директорию для результатов
            if not os.path.exists(self.settings['output_dir']):
                os.makedirs(self.settings['output_dir'])
                logger.info(f"Создана директория для результатов: {self.settings['output_dir']}")

            # Инициализируем анализатор
            self.analyzer = TextAnalyzer({
                'chunk_size': self.settings['chunk_size'],
                'language': self.settings['language'],
                'num_workers': self.settings['workers'],
                'similarity_threshold': self.settings['threshold']
            })

            # Добавляем примеры интентов
            self.analyzer.set_intent_examples({
                "мыслеформа_запроса_номера": [
                    "Чем больше покупок, тем лучше",
                    "Новые приобретения делают человека счастливее",
                    "Качество жизни определяется количеством вещей"
                ],
                "мыслеформа_развития": [
                    "Постоянное обучение - ключ к успеху",
                    "Саморазвитие важнее материальных ценностей",
                    "Инвестиции в знания приносят лучшие дивиденды"
                ]
            })

            # Инициализируем визуализатор
            self.visualizer = ResultVisualizer(self.settings['output_dir'])

            # Обрабатываем файл
            self.results = self.analyzer.analyze_file(
                self.current_file,
                language=self.settings['language']
            )

            # Создаем визуализации
            if 'word_frequencies' in self.results:
                self.visualizer.create_word_cloud(
                    dict(self.results['word_frequencies'][:self.settings['top_words']]),
                    title="Облако слов"
                )
                self.visualizer.create_bar_chart(
                    self.results['word_frequencies'][:20],
                    title="Топ-20 частых слов"
                )

            if 'intents' in self.results:
                self.visualizer.create_heatmap(
                    {intent['intent']: [intent['confidence']] 
                     for intent in self.results['intents']},
                    title="Схожесть намерений"
                )

            # Создаем HTML-отчет
            self.visualizer.create_html_report(self.results, analyzer=self.analyzer)
            
            print("\nАнализ завершен успешно!")
            print(f"Результаты сохранены в директории: {self.settings['output_dir']}")
            
        except Exception as e:
            print(f"Ошибка при выполнении анализа: {str(e)}")

    def show_results(self):
        """Просмотр результатов анализа"""
        if not self.results:
            print("Сначала выполните анализ")
            return

        print("\nРезультаты анализа:")
        print("-" * 30)
        
        if 'word_frequencies' in self.results:
            print("\nТоп-10 частых слов:")
            for word, freq in self.results['word_frequencies'][:10]:
                print(f"{word}: {freq}")
        
        if 'extracted_intents' in self.results:
            print("\nНайденные намерения:")
            for intent, matches in self.results['extracted_intents'].items():
                print(f"\n{intent}:")
                for match in matches[:3]:  # Показываем топ-3 совпадения
                    print(f"  - {match['text']} (схожесть: {match['similarity']:.2f})")
        
        if 'intents' in self.results:  # Для обратной совместимости
            print("\nНайденные намерения (старый формат):")
            for intent in self.results['intents']:
                print(f"  - {intent['intent']} (уверенность: {intent['confidence']:.2f})")
        
        print("\nПодробный отчет сохранен в HTML-файле в директории результатов")

    def main_menu(self):
        """Главное меню программы"""
        while True:
            self.print_header()
            print("1. Выбрать файл для анализа")
            print("2. Настроить параметры анализа")
            print("3. Запустить анализ")
            print("4. Просмотреть результаты")
            print("5. Выход")
            
            try:
                choice = int(input("\nВыберите действие: "))
                
                if choice == 1:
                    self.select_file()
                elif choice == 2:
                    self.change_setting()
                elif choice == 3:
                    self.run_analysis()
                elif choice == 4:
                    self.show_results()
                elif choice == 5:
                    print("До свидания!")
                    sys.exit(0)
                else:
                    print("Неверный выбор")
                
                input("\nНажмите Enter для продолжения...")
                
            except ValueError:
                print("Ошибка: введите число")
                input("\nНажмите Enter для продолжения...")

def main():
    """Основная функция"""
    cli = TextAnalysisCLI()
    cli.main_menu()

if __name__ == "__main__":
    main()
