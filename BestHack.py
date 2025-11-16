import pandas as pd
import Levenshtein
from typing import List, Dict, Any, Tuple
import sqlite3
from difflib import SequenceMatcher
import re
import time
from collections import defaultdict
import numpy as np

# Загружаем данные из вашего CSV файла с индексацией
def load_building_database():
    """Загружаем базу зданий из CSV файла с комплексной индексацией"""
    try:
        print("📥 Загрузка базы данных...")
        start_time = time.time()
        
        buildings_df = pd.read_csv('final_norm.csv')
        print(f"✅ База данных загружена: {len(buildings_df)} зданий")
        print(f"📊 Столбцы: {list(buildings_df.columns)}")
        
        # ============================================================================
        # 1. БАЗОВАЯ ИНДЕКСАЦИЯ - создаем нормализованные версии для поиска
        # ============================================================================
        print("\n🔧 Создание базовых индексов...")
        
        # Нормализация улиц
        buildings_df['street_normalized'] = buildings_df['street'].str.lower().str.strip()
        buildings_df['street_normalized'] = buildings_df['street_normalized'].fillna('')
        
        # Нормализация номеров домов
        buildings_df['housenumber_normalized'] = buildings_df['housenumber'].str.lower().str.strip()
        buildings_df['housenumber_normalized'] = buildings_df['housenumber_normalized'].fillna('')
        
        # Создаем полный адрес для отображения
        buildings_df['full_address'] = buildings_df.apply(
            lambda row: f"{row['street']}, {row['housenumber']}", axis=1
        )
        
        # ============================================================================
        # 2. ПРЕДВАРИТЕЛЬНАЯ ИНДЕКСАЦИЯ ДЛЯ БЫСТРОГО ПОИСКА
        # ============================================================================
        print("🔧 Создание предварительных индексов...")
        
        # Индекс по первым буквам улиц (для быстрого фильтра)
        buildings_df['street_first_letter'] = buildings_df['street_normalized'].str[0].fillna('')
        
        # Индекс по длине названия улицы
        buildings_df['street_length'] = buildings_df['street_normalized'].str.len()
        
        # Индекс по типам улиц (шоссе, улица, проспект и т.д.)
        def detect_street_type(street_name):
            street_lower = street_name.lower()
            if 'шоссе' in street_lower:
                return 'шоссе'
            elif 'проспект' in street_lower:
                return 'проспект'
            elif 'бульвар' in street_lower:
                return 'бульвар'
            elif 'проезд' in street_lower:
                return 'проезд'
            elif 'переулок' in street_lower:
                return 'переулок'
            elif 'набережная' in street_lower:
                return 'набережная'
            elif 'аллея' in street_lower:
                return 'аллея'
            else:
                return 'улица'
        
        buildings_df['street_type'] = buildings_df['street_normalized'].apply(detect_street_type)
        
        # ============================================================================
        # 3. СОЗДАЕМ СЛОВАРИ ДЛЯ БЫСТРОГО ДОСТУПА
        # ============================================================================
        print("🔧 Создание словарей для быстрого доступа...")
        
        # Словарь улиц -> список зданий на этой улице
        street_to_buildings = defaultdict(list)
        for idx, row in buildings_df.iterrows():
            street_to_buildings[row['street_normalized']].append(idx)
        
        # Словарь первых букв -> список улиц
        first_letter_to_streets = defaultdict(set)
        for street in buildings_df['street_normalized'].unique():
            if street:  # проверяем, что строка не пустая
                first_letter_to_streets[street[0]].add(street)
        
        # Словарь типов улиц -> список улиц
        street_type_to_streets = defaultdict(set)
        for _, row in buildings_df.iterrows():
            street_type_to_streets[row['street_type']].add(row['street_normalized'])
        
        # ============================================================================
        # 4. ГЕОГРАФИЧЕСКАЯ ИНДЕКСАЦИЯ (для пространственного поиска)
        # ============================================================================
        print("🔧 Создание географических индексов...")
        
        # Нормализуем координаты
        buildings_df['lat_normalized'] = pd.to_numeric(buildings_df['lat'], errors='coerce')
        buildings_df['lon_normalized'] = pd.to_numeric(buildings_df['lon'], errors='coerce')
        
        # Создаем географические зоны (квадраты 0.01 градуса ~ 1.1 км)
        buildings_df['geo_zone_lat'] = (buildings_df['lat_normalized'] * 100).astype(int)
        buildings_df['geo_zone_lon'] = (buildings_df['lon_normalized'] * 100).astype(int)
        buildings_df['geo_zone'] = buildings_df['geo_zone_lat'].astype(str) + '_' + buildings_df['geo_zone_lon'].astype(str)
        
        # Словарь географических зон -> список зданий
        geo_zone_to_buildings = defaultdict(list)
        for idx, row in buildings_df.iterrows():
            if not pd.isna(row['geo_zone']):
                geo_zone_to_buildings[row['geo_zone']].append(idx)
        
        # ============================================================================
        # 5. ИНДЕКСАЦИЯ ПО НОМЕРАМ ДОМОВ
        # ============================================================================
        print("🔧 Создание индексов номеров домов...")
        
        # Извлекаем числовую часть номера дома
        def extract_house_number(house_str):
            if pd.isna(house_str):
                return 0
            # Ищем первую последовательность цифр
            match = re.search(r'(\d+)', str(house_str))
            return int(match.group(1)) if match else 0
        
        buildings_df['house_number_numeric'] = buildings_df['housenumber'].apply(extract_house_number)
        
        # Словарь числовых номеров -> список зданий
        house_number_to_buildings = defaultdict(list)
        for idx, row in buildings_df.iterrows():
            house_number_to_buildings[row['house_number_numeric']].append(idx)
        
        # ============================================================================
        # 6. СОЗДАЕМ ОБЪЕКТ ИНДЕКСОВ ДЛЯ БЫСТРОГО ДОСТУПА
        # ============================================================================
        indices = {
            'street_to_buildings': dict(street_to_buildings),
            'first_letter_to_streets': dict(first_letter_to_streets),
            'street_type_to_streets': dict(street_type_to_streets),
            'geo_zone_to_buildings': dict(geo_zone_to_buildings),
            'house_number_to_buildings': dict(house_number_to_buildings)
        }
        
        loading_time = time.time() - start_time
        print(f"✅ Индексация завершена за {loading_time:.2f} секунд")
        
        return buildings_df, indices
        
    except FileNotFoundError:
        print("❌ Файл final_norm.csv не найден")
        return pd.DataFrame(), {}
    except Exception as e:
        print(f"❌ Ошибка при загрузке файла: {e}")
        return pd.DataFrame(), {}

# Загружаем базу данных с индексами
buildings_df, indices = load_building_database()

class BasicGeocoder:
    """Базовый алгоритм геокодирования - простой точный поиск"""

    def __init__(self, buildings_df):
        self.buildings_df = buildings_df.copy()
        self.street_replacements = {
            'ул.': 'улица',
            'пр.': 'проспект', 
            'ш.': 'шоссе',
            'наб.': 'набережная',
            'пер.': 'переулок',
            'б-р': 'бульвар',
            'пр-д': 'проезд',
            'пр-кт': 'проспект'
        }

    def normalize_address(self, address: str) -> Dict[str, str]:
        """Простая нормализация адреса"""
        address_lower = address.lower()

        # Замена сокращений
        for short, full in self.street_replacements.items():
            address_lower = address_lower.replace(short, full)

        # Предполагаем формат: город, улица, дом
        parts = [part.strip() for part in address_lower.split(',')]

        if len(parts) >= 3:
            return {
                'city': parts[0],
                'street': parts[1],
                'housenumber': parts[2]
            }
        elif len(parts) == 2:
            return {
                'city': 'москва',  # предполагаем Москву
                'street': parts[0],
                'housenumber': parts[1]
            }
        else:
            return {
                'city': 'москва',
                'street': parts[0],
                'housenumber': ''
            }

    def search(self, parsed_addr: Dict[str, str]) -> pd.DataFrame:
        """Точный поиск в базе данных"""
        if self.buildings_df.empty:
            return pd.DataFrame()

        # Поиск по улице и номеру дома
        mask = (
            (self.buildings_df['street_normalized'].str.lower() == parsed_addr['street'].lower())
        )

        if parsed_addr['housenumber']:
            number_mask = (
                (self.buildings_df['housenumber_normalized'].str.lower() == parsed_addr['housenumber'].lower())
            )
            mask = mask & number_mask

        return self.buildings_df[mask]

    def geocode(self, address: str) -> Dict[str, Any]:
        """Основной метод геокодирования"""
        if self.buildings_df.empty:
            return {
                "searched_address": address,
                "objects": [],
                "search_time": 0
            }

        start_time = time.time()

        parsed_addr = self.normalize_address(address)
        results = self.search(parsed_addr)

        objects = []
        for _, row in results.iterrows():
            objects.append({
                "locality": "Москва",
                "street": row['street'],
                "number": row['housenumber'],
                "lon": row['lon'],
                "lat": row['lat'],
                "score": 1.0,  # Все найденные считаются идеальными
                "full_address": row['full_address']
            })

        search_time = time.time() - start_time

        return {
            "searched_address": address,
            "objects": objects,
            "search_time": search_time
        }

class OptimizedGeocoder:
    """Оптимизированный геокодер с использованием предварительной индексации"""

    def __init__(self, buildings_df, indices):
        self.buildings_df = buildings_df.copy()
        self.indices = indices
        
        # Словари для быстрого доступа
        self.street_to_buildings = indices.get('street_to_buildings', {})
        self.first_letter_to_streets = indices.get('first_letter_to_streets', {})
        self.street_type_to_streets = indices.get('street_type_to_streets', {})
        self.geo_zone_to_buildings = indices.get('geo_zone_to_buildings', {})
        self.house_number_to_buildings = indices.get('house_number_to_buildings', {})
        
        self.street_replacements = {
            'ул.': 'улица', 'ул': 'улица',
            'пр.': 'проспект', 'пр': 'проспект',
            'ш.': 'шоссе', 'ш': 'шоссе',
            'наб.': 'набережная', 'наб': 'набережная',
            'пер.': 'переулок', 'пер': 'переулок',
            'б-р': 'бульвар', 'бр': 'бульвар',
            'пр-д': 'проезд', 'пр-кт': 'проспект',
            'проезд': 'проезд', 'аллея': 'аллея', 'ал.': 'аллея'
        }

    def normalize_street_name(self, street: str) -> str:
        """Нормализация названия улицы с использованием индексов"""
        if pd.isna(street):
            return ""

        street_lower = street.lower().strip()

        # Замена сокращений
        for short, full in self.street_replacements.items():
            street_lower = street_lower.replace(short, full)

        # Удаление лишних пробелов
        street_lower = re.sub(r'\s+', ' ', street_lower).strip()

        return street_lower

    def normalize_housenumber(self, housenumber: str) -> str:
        """Нормализация номера дома"""
        if pd.isna(housenumber):
            return ""

        normalized = str(housenumber).lower()
        normalized = normalized.replace('/', '').replace('\\', '').replace('с', 'с').replace('к', 'к')
        normalized = re.sub(r'\s+', '', normalized)

        return normalized

    def string_similarity(self, str1: str, str2: str) -> float:
        """Вычисление схожести строк с помощью SequenceMatcher"""
        return SequenceMatcher(None, str1, str2).ratio()

    def calculate_score(self, query_street: str, query_number: str,
                       db_street: str, db_number: str) -> float:
        """Расчет общего score для кандидата"""
        street_similarity = self.string_similarity(query_street, db_street)

        if query_number and db_number:
            number_similarity = self.string_similarity(query_number, db_number)
        else:
            number_similarity = 0.5

        # Улица важнее номера дома
        total_score = 0.7 * street_similarity + 0.3 * number_similarity
        return total_score

    def parse_address(self, address: str) -> Dict[str, str]:
        """Парсинг адреса с улучшенной логикой"""
        address_lower = address.lower().strip()
        
        # Удаляем лишние слова
        address_lower = re.sub(r'город\s+', '', address_lower)
        address_lower = re.sub(r'г\.?\s*', '', address_lower)
        address_lower = re.sub(r'дом\s+', '', address_lower)
        address_lower = re.sub(r'д\.?\s*', '', address_lower)
        address_lower = re.sub(r'улица\s+', '', address_lower)
        address_lower = re.sub(r'ул\.?\s*', '', address_lower)
        
        # Разделяем по запятым или пробелам
        if ',' in address_lower:
            parts = [part.strip() for part in address_lower.split(',')]
        else:
            parts = [part.strip() for part in address_lower.split() if part.strip()]
        
        city = 'москва'
        
        if len(parts) >= 3:
            if any(moscow_indicator in parts[0] for moscow_indicator in ['москва', 'мск']):
                city = 'москва'
                street = parts[1]
                housenumber = parts[2]
            else:
                street = parts[0]
                housenumber = parts[1]
        elif len(parts) == 2:
            street = parts[0]
            housenumber = parts[1]
        else:
            street = parts[0] if parts else ''
            housenumber = ''
        
        return {
            'city': city,
            'street': self.normalize_street_name(street),
            'housenumber': self.normalize_housenumber(housenumber)
        }

    def get_candidates_by_street(self, street_name: str) -> List[int]:
        """Получаем кандидатов по названию улицы с использованием индексов"""
        candidates = set()
        
        if not street_name:
            return list(candidates)
        
        # 1. Точное совпадение по индексу улиц
        if street_name in self.street_to_buildings:
            candidates.update(self.street_to_buildings[street_name])
        
        # 2. Поиск по первой букве (быстрая фильтрация)
        first_letter = street_name[0] if street_name else ''
        if first_letter in self.first_letter_to_streets:
            similar_streets = self.first_letter_to_streets[first_letter]
            for similar_street in similar_streets:
                if self.string_similarity(street_name, similar_street) > 0.6:
                    candidates.update(self.street_to_buildings.get(similar_street, []))
        
        return list(candidates)

    def geocode_optimized(self, address: str) -> Dict[str, Any]:
        """Оптимизированный метод геокодирования с использованием индексов"""
        if self.buildings_df.empty:
            return {
                "searched_address": address,
                "objects": [],
                "search_time": 0
            }

        start_time = time.time()
        
        # Парсинг входного адреса
        parsed_addr = self.parse_address(address)
        query_street = parsed_addr['street']
        query_number = parsed_addr['housenumber']

        # Получаем кандидатов с использованием индексов
        candidate_indices = self.get_candidates_by_street(query_street)
        
        if not candidate_indices:
            # Если не нашли по индексу, ищем по всему датасету (медленнее)
            candidates_df = self.buildings_df.copy()
        else:
            # Фильтруем только кандидатов
            candidates_df = self.buildings_df.iloc[candidate_indices].copy()

        if candidates_df.empty:
            return {
                "searched_address": address,
                "objects": [],
                "search_time": time.time() - start_time
            }

        # Расчет score для каждого кандидата
        candidates_df['score'] = candidates_df.apply(
            lambda row: self.calculate_score(
                query_street, query_number,
                row['street_normalized'], row['housenumber_normalized']
            ), axis=1
        )

        # Сортировка по убыванию score и выбор топ-5
        top_candidates = candidates_df.nlargest(5, 'score')

        objects = []
        for _, row in top_candidates.iterrows():
            objects.append({
                "locality": "Москва",
                "street": row['street'],
                "number": row['housenumber'],
                "lon": row['lon'],
                "lat": row['lat'],
                "score": round(row['score'], 4),
                "full_address": row['full_address']
            })

        search_time = time.time() - start_time
        
        return {
            "searched_address": address,
            "objects": objects,
            "search_time": search_time,
            "candidates_count": len(candidate_indices)
        }

def display_basic_algorithm_results(query: str, results: Dict[str, Any], test_number: int = None, expected_address: str = None):
    """Отображает результаты базового алгоритма"""
    print(f"\n{'='*80}")
    print("🔹 БАЗОВЫЙ АЛГОРИТМ (точный поиск)")
    print(f"{'='*80}")
    
    if test_number is not None:
        print(f"Тест {test_number}: '{query}'")
    else:
        print(f"Запрос: '{query}'")
    
    if expected_address:
        print(f"Ожидается: {expected_address}")
    
    if not results['objects']:
        print("Найдено: ❌ адрес не найден")
        print("Причина: требуется точное совпадение улицы и номера дома")
        return
    
    # Базовый алгоритм возвращает все результаты с score 1.0
    best_result = results['objects'][0]
    print(f"Найдено: {best_result['street']}, {best_result['number']}")
    print(f"Score: {best_result['score']} (фиксированный для точных совпадений)")
    print(f"Координаты: {best_result['lat']:.6f}, {best_result['lon']:.6f}")
    
    # Проверка совпадения с ожидаемым адресом
    if expected_address:
        expected_normalized = expected_address.lower().replace('ул.', 'улица').replace(' ', '')
        found_normalized = f"{best_result['street']},{best_result['number']}".lower().replace(' ', '')
        
        if expected_normalized in found_normalized or found_normalized in expected_normalized:
            print("Совпадение: ✅ ТОЧНОЕ")
        else:
            print("Совпадение: ❌")
    
    # Другие кандидаты (если есть)
    if len(results['objects']) > 1:
        print(f"Всего точных совпадений: {len(results['objects'])}")
        print("Другие кандидаты:")
        for i, candidate in enumerate(results['objects'][1:], 1):
            print(f"  - {candidate['street']}, {candidate['number']} (score: {candidate['score']})")
    
    if 'search_time' in results:
        print(f"⏱️  Время поиска: {results['search_time']:.4f} сек")

def display_advanced_algorithm_results(query: str, results: Dict[str, Any], test_number: int = None, expected_address: str = None):
    """Отображает результаты улучшенного алгоритма"""
    print(f"\n{'='*80}")
    print("🔸 УЛУЧШЕННЫЙ АЛГОРИТМ (нечеткий поиск)")
    print(f"{'='*80}")
    
    if test_number is not None:
        print(f"Тест {test_number}: '{query}'")
    else:
        print(f"Запрос: '{query}'")
    
    if expected_address:
        print(f"Ожидается: {expected_address}")
    
    if not results['objects']:
        print("Найдено: ❌ адрес не найден")
        return
    
    # Лучший результат
    best_result = results['objects'][0]
    print(f"Найдено: {best_result['street']}, {best_result['number']}")
    print(f"Score: {best_result['score']}")
    print(f"Координаты: {best_result['lat']:.6f}, {best_result['lon']:.6f}")
    
    # Проверка совпадения с ожидаемым адресом
    if expected_address:
        expected_normalized = expected_address.lower().replace('ул.', 'улица').replace(' ', '')
        found_normalized = f"{best_result['street']},{best_result['number']}".lower().replace(' ', '')
        
        similarity = SequenceMatcher(None, expected_normalized, found_normalized).ratio()
        
        if similarity > 0.9:
            print("Совпадение: ✅ ТОЧНОЕ")
        elif similarity > 0.7:
            print("Совпадение: ✅ ХОРОШЕЕ")
        elif similarity > 0.5:
            print("Совпадение: ⚠️  ЧАСТИЧНОЕ")
        else:
            print("Совпадение: ❌")
    
    # Другие кандидаты (если есть)
    if len(results['objects']) > 1:
        print("Другие кандидаты:")
        for i, candidate in enumerate(results['objects'][1:], 1):
            print(f"  - {candidate['street']}, {candidate['number']} (score: {candidate['score']})")
    
    if 'search_time' in results:
        print(f"⏱️  Время поиска: {results['search_time']:.4f} сек")
    if 'candidates_count' in results:
        print(f"🔍 Кандидатов рассмотрено: {results['candidates_count']}")

def display_comparison(basic_results: Dict[str, Any], advanced_results: Dict[str, Any]):
    """Сравнивает результаты двух алгоритмов"""
    print(f"\n{'='*80}")
    print("📊 СРАВНЕНИЕ АЛГОРИТМОВ")
    print(f"{'='*80}")
    
    basic_found = len(basic_results['objects'])
    advanced_found = len(advanced_results['objects'])
    
    print(f"🔹 Базовый алгоритм: {basic_found} результатов")
    print(f"🔸 Улучшенный алгоритм: {advanced_found} результатов")
    
    if basic_found > 0 and advanced_found > 0:
        best_basic = basic_results['objects'][0]
        best_advanced = advanced_results['objects'][0]
        
        print(f"\n🎯 Лучшие результаты:")
        print(f"   Базовый: {best_basic['street']}, {best_basic['number']} (score: {best_basic['score']})")
        print(f"   Улучшенный: {best_advanced['street']}, {best_advanced['number']} (score: {best_advanced['score']})")
        
        # Сравнение времени
        basic_time = basic_results.get('search_time', 0)
        advanced_time = advanced_results.get('search_time', 0)
        
        print(f"\n⏱️  Время выполнения:")
        print(f"   Базовый: {basic_time:.4f} сек")
        print(f"   Улучшенный: {advanced_time:.4f} сек")
        
        if advanced_time > 0:
            speed_ratio = basic_time / advanced_time
def demo_mode_both_algorithms():
    """Демонстрационный режим с обоими алгоритмами"""
    print("🎯 ДЕМОНСТРАЦИОННЫЙ РЕЖИМ - СРАВНЕНИЕ АЛГОРИТМОВ")
    
    basic_geocoder = BasicGeocoder(buildings_df)
    advanced_geocoder = OptimizedGeocoder(buildings_df, indices)
    
    # Тестовые примеры с ожидаемыми результатами
    test_cases = [
        {
            'query': 'Москва, смольная ул., 24г с.4',
            'expected': 'смольная ул., 24г с.4'
        },
        {
            'query': 'дмитровское шоссе, 165б',
            'expected': 'дмитровское шоссе, 165б'
        },
        {
            'query': 'правобережная улица, 1б',
            'expected': 'правобережная улица, 1б'
        },
        {
            'query': 'аэродромная ул. 9',
            'expected': 'аэродромная ул., 9'
        },
        {
            'query': 'туристская улица, 2к5',
            'expected': 'туристская улица, 2к5'
        },
        {
            'query': 'несуществующая улица, 123',
            'expected': 'несуществующая улица, 123'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        query = test_case['query']
        expected = test_case['expected']
        
        print(f"\n{'#'*100}")
        print(f"🎯 ТЕСТ {i}: '{query}'")
        print(f"{'#'*100}")
        
        # Запуск обоих алгоритмов
        basic_results = basic_geocoder.geocode(query)
        advanced_results = advanced_geocoder.geocode_optimized(query)
        
        # Вывод результатов
        display_basic_algorithm_results(query, basic_results, test_number=i, expected_address=expected)
        display_advanced_algorithm_results(query, advanced_results, test_number=i, expected_address=expected)
        
        # Сравнение
        display_comparison(basic_results, advanced_results)
        
        if i < len(test_cases):
            input("\n⏎ Нажмите Enter для следующего теста...")

def interactive_geocoding_both_algorithms():
    """Интерактивный режим с обоими алгоритмами"""
    print("🚀 ИНТЕРАКТИВНЫЙ ГЕОКОДЕР - СРАВНЕНИЕ АЛГОРИТМОВ")
    print(f"📊 Загружено зданий: {len(buildings_df)}")
    
    print("\n💡 Примеры запросов:")
    print("   - Москва, смольная ул., 24г с.4")
    print("   - дмитровское шоссе, 165б")
    print("   - правобережная ул, 1б") 
    print("   - аэродромная улица 9")
    print("   - туристская ул, 2к5")
    print("   - несуществующая улица, 123 (для теста)")
    print("   - выход - для завершения работы")
    print("-" * 50)
    
    basic_geocoder = BasicGeocoder(buildings_df)
    advanced_geocoder = OptimizedGeocoder(buildings_df, indices)
    
    while True:
        try:
            query = input("\n📍 Введите адрес для поиска: ").strip()
            
            if query.lower() in ['выход', 'exit', 'quit', 'q']:
                print("👋 До свидания!")
                break
            
            if not query:
                print("⚠️  Пожалуйста, введите адрес")
                continue
            
            print("\n" + "🔄 Поиск..." + "🔍" * 3)
            
            # Запуск обоих алгоритмов
            basic_results = basic_geocoder.geocode(query)
            advanced_results = advanced_geocoder.geocode_optimized(query)
            
            # Вывод результатов
            display_basic_algorithm_results(query, basic_results)
            display_advanced_algorithm_results(query, advanced_results)
            display_comparison(basic_results, advanced_results)
            
            print(f"\n💡 Продолжайте вводить адреса или введите 'выход' для завершения")
            
        except KeyboardInterrupt:
            print("\n\n👋 До свидания!")
            break
        except Exception as e:
            print(f"❌ Ошибка при обработке запроса: {e}")

def show_indexing_info():
    """Показывает информацию об индексации"""
    if not indices:
        print("❌ Индексы не созданы")
        return
    
    print(f"\n{'='*50}")
    print("📊 ИНФОРМАЦИЯ ОБ ИНДЕКСАЦИИ")
    print(f"{'='*50}")
    
    print(f"🔤 Уникальных улиц: {len(indices['street_to_buildings'])}")
    print(f"🔠 Первых букв улиц: {len(indices['first_letter_to_streets'])}")
    print(f"🏷️  Типов улиц: {len(indices['street_type_to_streets'])}")
    print(f"🌍 Географических зон: {len(indices['geo_zone_to_buildings'])}")
    print(f"🏠 Уникальных номеров домов: {len(indices['house_number_to_buildings'])}")
    
    # Статистика по типам улиц
    print(f"\n📈 Статистика по типам улиц:")
    for street_type, streets in indices['street_type_to_streets'].items():
        print(f"   - {street_type}: {len(streets)} улиц")

def benchmark_search():
    """Тестирование производительности поиска"""
    print("\n🎯 ТЕСТИРОВАНИЕ ПРОИЗВОДИТЕЛЬНОСТИ")
    
    test_queries = [
        'дмитровское шоссе, 165б',
        'правобережная ул, 1б',
        'аэродромная улица 9',
        'туристская ул, 2к5',
        'бульвар яна райниса, 19к2'
    ]
    
    basic_geocoder = BasicGeocoder(buildings_df)
    advanced_geocoder = OptimizedGeocoder(buildings_df, indices)
    
    basic_times = []
    advanced_times = []
    
    for query in test_queries:
        # Базовый алгоритм
        start_time = time.time()
        basic_results = basic_geocoder.geocode(query)
        basic_time = time.time() - start_time
        basic_times.append(basic_time)
        
        # Улучшенный алгоритм
        start_time = time.time()
        advanced_results = advanced_geocoder.geocode_optimized(query)
        advanced_time = time.time() - start_time
        advanced_times.append(advanced_time)
        
        print(f"🔍 '{query}':")
        print(f"   Базовый: {basic_time:.4f} сек, найдено: {len(basic_results['objects'])}")
        print(f"   Улучшенный: {advanced_time:.4f} сек, найдено: {len(advanced_results['objects'])}")
    
    print(f"\n📊 СТАТИСТИКА ПРОИЗВОДИТЕЛЬНОСТИ:")
    print(f"   Базовый алгоритм:")
    print(f"      - Среднее: {np.mean(basic_times):.4f} сек")
    print(f"      - Минимум: {min(basic_times):.4f} сек") 
    print(f"      - Максимум: {max(basic_times):.4f} сек")
    print(f"   Улучшенный алгоритм:")
    print(f"      - Среднее: {np.mean(advanced_times):.4f} сек")
    print(f"      - Минимум: {min(advanced_times):.4f} сек") 
    print(f"      - Максимум: {max(advanced_times):.4f} сек")

def main():
    """Главное меню программы"""
    if buildings_df.empty:
        print("❌ Не удалось загрузить базу данных зданий")
        return
    
    print("🏙️  ГЕОКОДЕР МОСКВЫ - СРАВНЕНИЕ АЛГОРИТМОВ")
    print(f"📊 База данных: {len(buildings_df)} зданий")
    print(f"🔧 Индексов создано: {len(indices)} типов")
    
    while True:
        print("\n" + "="*60)
        print("🎮 ВЫБЕРИТЕ РЕЖИМ РАБОТЫ:")
        print("1 - 🔍 Интерактивный поиск (оба алгоритма)")
        print("2 - 🎯 Демонстрационный режим (оба алгоритма)")
        print("3 - 📊 Информация об индексации")
        print("4 - 🎯 Тестирование производительности")
        print("5 - 📋 Информация о базе данных")
        print("6 - 🚪 Выход")
        print("="*60)
        
        choice = input("Ваш выбор (1-6): ").strip()
        
        if choice == '1':
            interactive_geocoding_both_algorithms()
        elif choice == '2':
            demo_mode_both_algorithms()
        elif choice == '3':
            show_indexing_info()
        elif choice == '4':
            benchmark_search()
        elif choice == '5':
            show_database_info()
        elif choice == '6':
            print("👋 До свидания!")
            break
        else:
            print("❌ Неверный выбор. Пожалуйста, введите 1-6")

def show_database_info():
    """Показывает информацию о базе данных"""
    print(f"\n{'='*50}")
    print("📊 ИНФОРМАЦИЯ О БАЗЕ ДАННЫХ")
    print(f"{'='*50}")
    print(f"🏢 Всего зданий: {len(buildings_df)}")
    print(f"🛣️  Уникальных улиц: {buildings_df['street'].nunique()}")
    
    # Географический охват
    print(f"\n🌍 Географический охват:")
    print(f"   Широта: {buildings_df['lat'].min():.6f} - {buildings_df['lat'].max():.6f}")
    print(f"   Долгота: {buildings_df['lon'].min():.6f} - {buildings_df['lon'].max():.6f}")

# Запуск программы
if __name__ == "__main__":
    main()
