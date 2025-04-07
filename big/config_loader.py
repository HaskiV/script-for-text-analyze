import json
import yaml
import os

class ConfigLoader:
    """Утилитный класс для загрузки конфигурации из JSON или YAML файлов"""
    
    @staticmethod
    def load_config(config_path):
        """
        Загрузка конфигурации из JSON или YAML файла
        
        Args:
            config_path (str): Путь к файлу конфигурации
            
        Returns:
            dict: Словарь конфигурации
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")
            
        file_ext = os.path.splitext(config_path)[1].lower()
        
        try:
            if file_ext in ['.json']:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            elif file_ext in ['.yml', '.yaml']:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            else:
                raise ValueError(f"Неподдерживаемый формат файла конфигурации: {file_ext}")
                
            return config
        except Exception as e:
            raise Exception(f"Ошибка загрузки конфигурации из {config_path}: {str(e)}")
    
    @staticmethod
    def get_intent_examples(config):
        """
        Извлечение примеров намерений из конфигурации
        
        Args:
            config (dict): Словарь конфигурации
            
        Returns:
            dict: Словарь примеров намерений
        """
        intents = {}
        
        if 'intents' in config:
            for intent in config['intents']:
                if 'name' in intent and 'examples' in intent:
                    intents[intent['name']] = intent['examples']
        
        return intents
    
    @staticmethod
    def get_analysis_settings(config):
        """
        Извлечение настроек анализа из конфигурации
        
        Args:
            config (dict): Словарь конфигурации
            
        Returns:
            dict: Словарь настроек анализа
        """
        settings = {
            'chunk_size': 5000,
            'language': 'auto',
            'num_workers': 4,
            'similarity_threshold': 0.6,
            'top_n_words': 100,
            'output_dir': 'output'
        }
        
        if 'settings' in config:
            for key in settings:
                if key in config['settings']:
                    settings[key] = config['settings'][key]
        
        return settings


# Пример файла конфигурации (формат YAML)
EXAMPLE_CONFIG = """
# Конфигурация анализа текста
settings:
  chunk_size: 1000
  language: auto
  num_workers: 4
  similarity_threshold: 0.7
  top_n_words: 100
  output_dir: ./results

# Определения намерений
intents:
  - name: question
    examples:
      - "Как использовать эту функцию?"
      - "Какова цена этого продукта?"
      - "Не могли бы вы объяснить, как это работает?"
      - "How can I use this feature?"
      - "What is the price of this product?"
      - "Could you explain how this works?"
      
  - name: complaint
    examples:
      - "Это не работает должным образом"
      - "У меня проблема с этим продуктом"
      - "Я хочу вернуть этот товар, потому что он неисправен"
      - "I am not satisfied with your service"
      - "This product doesn't work as advertised"
      - "I want to file a complaint about the quality"
      
  - name: suggestion
    examples:
      - "Было бы лучше, если бы вы добавили"
      - "Я предлагаю улучшить"
      - "Вам следует рассмотреть возможность добавления"
      - "It would be better if you could add"
      - "I suggest improving"
      - "You should consider adding"

# Файлы для обработки
files:
  - path/to/feedback.txt
  - path/to/reviews.csv
  - path/to/customer_emails.xlsx
"""

# Пример создания файла конфигурации по умолчанию
def create_default_config(path="config.yml"):
    """Создание файла конфигурации по умолчанию"""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(EXAMPLE_CONFIG)
    print(f"Конфигурация по умолчанию создана в: {path}")
