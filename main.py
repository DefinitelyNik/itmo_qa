import requests
from bs4 import BeautifulSoup
import time

# Определение целевых URL
URLS = {
    "Искусственный интеллект": "https://abit.itmo.ru/program/master/ai",
    "AI продукт": "https://abit.itmo.ru/program/master/ai_product",
}

# Заголовки для имитации запроса от браузера (может помочь избежать блокировок)
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36'
}

def parse_page(url, program_name):
    """
    Парсит отдельную страницу программы.
    :param url: URL страницы программы.
    :param program_name: Название программы для заголовка вывода.
    :return: Словарь с названием программы и извлеченным текстом.
    """
    print(f"Начинаем парсинг страницы: {program_name} ({url})")

    try:
        # Отправка GET-запроса
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status() # Проверка на успешный код ответа (200)

        # Создание объекта BeautifulSoup для разбора HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # Инициализация переменной для хранения результата
        parsed_text = []

        # --- Извлечение основного содержимого ---
        # На основе анализа структуры, основной контент находится внутри <main> или <div class="program-page">
        # Мы будем искать заголовки и связанный с ними текст

        # Определяем корневой элемент для поиска контента
        # Иногда полезно сузить область поиска
        content_area = soup.find('main') or soup.find('div', class_='program-page') or soup

        # Если область контента найдена, начинаем парсинг внутри нее
        if content_area:
            # Ищем все заголовки h1, h2, h3, h4, h5, h6
            # Также ищем параграфы <p> и списки <ul>, <ol> как основной текст
            # Можно адаптировать список тегов в зависимости от структуры
            elements = content_area.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'div'])

            for element in elements:
                # Проверяем, является ли элемент заголовком
                if element.name.startswith('h'):
                    # Получаем текст заголовка
                    header_text = element.get_text(strip=True)
                    if header_text: # Добавляем только непустые заголовки
                         # Добавляем заголовок с маркером для удобства (можно убрать при необходимости)
                        parsed_text.append(f"[{element.name.upper()}] {header_text}")

                # Проверяем на параграф
                elif element.name == 'p':
                    text = element.get_text(strip=True)
                    if text:
                        parsed_text.append(text)

                # Проверяем на списки
                elif element.name in ['ul', 'ol']:
                    list_items = []
                    for li in element.find_all('li'):
                        li_text = li.get_text(strip=True)
                        if li_text:
                            # Добавляем отступ для визуального отделения пунктов списка
                            list_items.append(f"  - {li_text}")
                    if list_items:
                        parsed_text.extend(list_items)

                # Проверяем div на наличие текста (например, описание программы)
                # Это может быть менее точным, поэтому добавляем условие на наличие текста
                # и отсутствие дочерних элементов, которые мы уже обработали
                elif element.name == 'div':
                     # Пример: описание программы в div без класса или с определенным классом
                     # Этот блок может потребовать адаптации
                     # Проверим, есть ли у div значимый текст напрямую
                     direct_text = element.get_text(strip=True)
                     if direct_text and len(direct_text) > 50: # Минимальная длина для значимости
                         # Избегаем дублирования, если внутри есть уже обработанные теги
                         # Простая проверка: если внутри много дочерних элементов, пропускаем
                         if len(element.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol'])) < 2:
                             parsed_text.append(direct_text)

        print(f"Парсинг страницы '{program_name}' завершен.")
        return {"program_name": program_name, "url": url, "content": parsed_text}

    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе к {url}: {e}")
        return {"program_name": program_name, "url": url, "content": [f"Ошибка: Не удалось получить страницу. {e}"]}
    except Exception as e:
        print(f"Неожиданная ошибка при парсинге {url}: {e}")
        return {"program_name": program_name, "url": url, "content": [f"Ошибка: Неожиданная ошибка при парсинге. {e}"]}


def main():
    """Главная функция для запуска парсинга всех URL."""
    all_data = []

    for name, url in URLS.items():
        # Добавляем небольшую задержку между запросами, чтобы не нагружать сервер
        time.sleep(1)
        data = parse_page(url, name)
        all_data.append(data)

    # Вывод результатов
    print("\n" + "="*50)
    print("РЕЗУЛЬТАТЫ ПАРСИНГА")
    print("="*50 + "\n")

    for data in all_data:
        print(f"--- Программа: {data['program_name']} ---")
        print(f"URL: {data['url']}\n")
        for line in data['content']:
            print(line)
        print("\n" + "-"*30 + "\n")

if __name__ == "__main__":
    main()

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import pipeline, AutoTokenizer

# --- 1. Данные ---
# Ваши тексты из парсинга, организованные по программам
all_data = {}

for name, url in URLS.items():
    # Добавляем небольшую задержку между запросами, чтобы не нагружать сервер
    time.sleep(1)
    data = parse_page(url, name)
    all_data[data["program_name"]] = '. '.join(data['content'])

PROGRAM_DATA = all_data

# --- 2. Предварительная обработка текста ---
def preprocess_text(text):
    """Удаляет markdown заголовки и изображения, оставляя основной текст."""
    # Удаление заголовков markdown
    text = re.sub(r'#+\s*', '', text)
    # Удаление строк с изображениями
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    # Удаление строк с иконками (предполагая, что они тоже в формате ![...](...))
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    # Разделение на предложения (простой способ)
    sentences = re.split(r'[.!?]+', text)
    # Очистка и фильтрация предложений
    cleaned_sentences = [s.strip() for s in sentences if s.strip()]
    return cleaned_sentences

# --- 3. Подготовка данных для поиска ---
# Словарь для хранения обработанных фрагментов и их источников
documents = []
document_metadata = [] # Хранит (program_name, original_section_header)

for program_name, full_text in PROGRAM_DATA.items():
    # Предположим, что разделы разделены двойным переносом строки \n\n
    # Или мы можем разделить по заголовкам, но для простоты используем предложения
    # Разбиваем весь текст на предложения
    sentences = preprocess_text(full_text)

    # Группируем предложения в более крупные фрагменты (например, по 3-5 предложений)
    # Это поможет получить более содержательный контекст для генерации
    chunk_size = 4
    for i in range(0, len(sentences), chunk_size):
        chunk = " ".join(sentences[i:i + chunk_size])
        if chunk: # Добавляем только непустые фрагменты
            documents.append(chunk)
            # Метаданные: для простоты указываем программу. Можно уточнить раздел.
            document_metadata.append(program_name)

# --- 4. Векторизация и поиск ---
# Создаем TF-IDF векторизатор
vectorizer = TfidfVectorizer()
# Векторизуем все документы
doc_vectors = vectorizer.fit_transform(documents)

def retrieve_relevant_docs(query, top_k=3):
    """Находит топ-K наиболее релевантных фрагментов документа для запроса."""
    # Векторизуем запрос
    query_vector = vectorizer.transform([query])
    # Вычисляем косинусное сходство между запросом и всеми документами
    similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    # Получаем индексы топ-K документов с наибольшим сходством
    top_indices = similarities.argsort()[-top_k:][::-1]

    # Формируем список кортежей (фрагмент, оценка сходства, источник)
    results = []
    for idx in top_indices:
        if similarities[idx] > 0: # Игнорируем нулевые совпадения
            results.append((documents[idx], similarities[idx], document_metadata[idx]))
    return results

# --- 5. Генерация ответа ---
# Загружаем модель для генерации текста. Используем относительно легкую модель.
# ruGPT3 может быть тяжелой, поэтому используем mT5 или ruT5, если доступны.
# Для демонстрации используем multilingual модель.
# Примечание: Загрузка модели может занять время при первом запуске.

# Выбираем модель. ruT5-base может быть хорошим выбором, но для демонстрации
# используем mT5-small, который поддерживает русский и загружается быстрее.
# model_name = "ai-forever/ruT5-base" # Требует установки SentencePiece
model_name = "google/mt5-small" # Хорошо работает с русским

# Создаем пайплайн для суммаризации (можно адаптировать для question-answering)
# Используем summarization как промежуточный вариант для генерации ответа на основе контекста
# Для более точного QA можно использовать модель типа ruBert для извлечения, а затем генерацию.
# Но для простоты используем генерацию.

# Пайплайн для генерации текста (summarization)
print("Загрузка модели для генерации (это может занять минуту)...")
qa_pipeline = pipeline(
    "text2text-generation",
    model=model_name,
    tokenizer=model_name,
    device=0 if torch.cuda.is_available() else -1 # Использовать GPU, если доступен
)
print("Модель загружена.")

def generate_answer(query, context_snippets):
    """Генерирует ответ на основе запроса и найденного контекста."""
    # Объединяем топ фрагменты в один контекст
    combined_context = "\n".join([snippet for snippet, _, _ in context_snippets])

    # Формируем входной промпт для модели
    # Формат промпта важен для получения релевантного ответа
    prompt = (
        f"Используя следующую информацию:\n{combined_context}\n\n"
        f"Ответь на вопрос: {query}\n"
        f"Ответ:"
    )

    # print(f"DEBUG: Prompt sent to model:\n{prompt}\n") # Для отладки

    # Генерируем ответ
    # Параметры можно настроить для лучшего качества
    outputs = qa_pipeline(
        prompt,
        max_length=200, # Максимальная длина генерируемого ответа
        min_length=10,  # Минимальная длина
        do_sample=True, # Включает сэмплирование для разнообразия
        truncation=True, # Обрезает вход, если он слишком длинный
        temperature=0.7, # Контролирует случайность (0 - детерминировано, выше - креативнее)
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2 # Штраф за повторения
    )

    # Извлекаем сгенерированный текст
    answer = outputs[0]['generated_text']

    # Иногда модель повторяет часть промпта, пытаемся извлечь только ответ
    if "Ответ:" in answer:
        answer = answer.split("Ответ:", 1)[1].strip()
    return answer


# --- 6. Основной цикл взаимодействия ---
def main():
    """Главная функция для запуска интерактивного Q&A."""
    print("\nДобро пожаловать в Q&A RAG по программам ИТМО!")
    print("Вы можете задавать вопросы о программах 'Искусственный интеллект' и 'AI продукт'.")
    print("Введите 'выход' или 'quit', чтобы завершить работу.\n")

    while True:
        query = input("Ваш вопрос: ").strip()
        if query.lower() in ['выход', 'quit', 'exit']:
            print("До свидания!")
            break
        if not query:
            continue

        print("Поиск релевантной информации...")
        # 1. Поиск релевантных фрагментов
        relevant_docs = retrieve_relevant_docs(query, top_k=3)

        if not relevant_docs:
             print("Извините, я не нашел релевантной информации для вашего вопроса.\n")
             continue

        print("Генерация ответа...")
        # 2. Генерация ответа на основе найденного
        answer = generate_answer(query, relevant_docs)

        # 3. Вывод результата
        print(f"\nОтвет: {answer}")
        # print("\nИсточники информации:")
        # for i, (doc, score, source) in enumerate(relevant_docs):
        #     print(f"  [{i+1}] (Сходство: {score:.4f}, Источник: {source}) {doc[:100]}...")
        print("-" * 20 + "\n")

if __name__ == "__main__":
    main()