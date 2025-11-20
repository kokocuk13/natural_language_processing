import os
from gensim.models import KeyedVectors

def load_model(filename):
    """Загрузка векторной модели из файла"""
    if not os.path.exists(filename):
        print(f"Файл {filename} не найден")
        return None
    
    try:
        model = KeyedVectors.load_word2vec_format(filename)
        print(f"Модель загружена. Слов: {len(model.key_to_index)}, Размерность: {model.vector_size}")
        return model
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        return None

def main():
    # Загрузка модели
    model = load_model("cbow.txt")
    if model is None:
        return
    
    # Целевые слова
    target_word1 = "концерт_NOUN"
    target_word2 = "балетмейстер_NOUN"
    
    print(f"Поиск линейной комбинации для: {target_word1} и {target_word2}")
    print("=" * 60)
    
    # Берем только существительные и ограниченное количество слов
    all_words = list(model.key_to_index.keys())
    
    # Фильтруем только существительные (NOUN) и берем первые 5000
    noun_words = [word for word in all_words if '_NOUN' in word][:5000]
    
    print(f"Используем {len(noun_words)} существительных для поиска")
    
    found_solution = False
    
    # Ищем среди ближайших слов к целевым
    similar_to_target1 = model.most_similar(target_word1, topn=100)
    similar_to_target2 = model.most_similar(target_word2, topn=100)
    
    # Собираем кандидатов из ближайших слов
    candidates = []
    for word, score in similar_to_target1 + similar_to_target2:
        if word not in [target_word1, target_word2] and word not in candidates:
            candidates.append(word)
    
    print(f"Тестируем комбинации из {len(candidates)} ближайших слов")
    
    # Тестируем комбинации из ближайших слов
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            word1 = candidates[i]
            word2 = candidates[j]
            
            try:
                # ЛИНЕЙНАЯ КОМБИНАЦИЯ: word1 + word2
                result = model.most_similar(positive=[word1, word2], topn=12)
                words_found = [word for word, score in result]
                
                # Проверяем есть ли оба целевых слова в топ-10
                if target_word1 in words_found and target_word2 in words_found:
                    pos1 = words_found.index(target_word1) + 1
                    pos2 = words_found.index(target_word2) + 1
                    
                    if pos1 <= 10 and pos2 <= 10:
                        print(f"\nНАЙДЕНА ЛИНЕЙНАЯ КОМБИНАЦИЯ:")
                        print(f"ИСХОДНАЯ ЛИНЕЙНАЯ КОМБИНАЦИЯ: {word1} + {word2}")
                        print(f"РЕЗУЛЬТАТ: {target_word1} - позиция {pos1}, {target_word2} - позиция {pos2}")
                        
                        print(f"\n10 ближайших слов к комбинации '{word1} + {word2}':")
                        for idx, (word, score) in enumerate(result[:10], 1):
                            marks = []
                            if word == target_word1: marks.append("концерт")
                            if word == target_word2: marks.append("балетмейстер")
                            mark_str = " <- " + ", ".join(marks) if marks else ""
                            print(f"{idx:2d}. {word}: {score:.4f}{mark_str}")
                        
                        found_solution = True
                        break
                        
            except:
                continue
        
        if found_solution:
            break
    
    if not found_solution:
        print("Не найдено подходящей линейной комбинации")

if __name__ == "__main__":
    main()
