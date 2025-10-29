import os
from gensim.models import KeyedVectors
import random

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

def find_similar_words(model, positive_words, negative_words=None, topn=15):
    """Нахождение ближайших слов к результату векторных операций"""
    if negative_words is None:
        negative_words = []
    
    try:
        result = model.most_similar(positive=positive_words, negative=negative_words, topn=topn)
        return result
    except Exception as e:
        return []

def main():
    # Загрузка модели
    model = load_model("cbow.txt")
    if model is None:
        return
    
    # Целевые слова
    target_word1 = "концерт_NOUN"
    target_word2 = "балетмейстер_NOUN"
    
    print(f"Поиск комбинации для: {target_word1} и {target_word2}")
    print("=" * 60)
    
    # Найдем ближайшие слова к каждому целевому слову
    similar_to_concert = model.most_similar(target_word1, topn=30)
    similar_to_balletmaster = model.most_similar(target_word2, topn=30)
    
    print("Ближайшие к 'концерт_NOUN':")
    concert_candidates = []
    for word, score in similar_to_concert[:15]:
        print(f"  {word}: {score:.4f}")
        concert_candidates.append(word)
    
    print("\nБлижайшие к 'балетмейстер_NOUN':")
    balletmaster_candidates = []
    for word, score in similar_to_balletmaster[:15]:
        print(f"  {word}: {score:.4f}")
        balletmaster_candidates.append(word)
    
    # Тестируем разные стратегии
    
    # Способ: Слова близкие к обоим целевым словам
    print("\n" + "="*50)
    print("Способ: слова близкие к обоим целевым")
    print("="*50)
    
    found_solution = False
    
    for word1 in concert_candidates:
        for word2 in balletmaster_candidates:
            if word1 == word2:
                continue
                
            # Тестируем сложение
            similar_words = find_similar_words(model, [word1, word2], topn=15)
            if not similar_words:
                continue
                
            words_list = [word for word, score in similar_words]
            
            pos1 = words_list.index(target_word1) + 1 if target_word1 in words_list else 0
            pos2 = words_list.index(target_word2) + 1 if target_word2 in words_list else 0
            
            if pos1 <= 10 and pos2 <= 10:
                print(f"\nКомбинация найдена")
                print(f"Комбинация: {word1} + {word2}")
                print(f"Позиции: {target_word1} - {pos1}, {target_word2} - {pos2}")
                print("\n10 ближайших слов:")
                for idx, (word, score) in enumerate(similar_words[:10], 1):
                    marker = ""
                    if word == target_word1:
                        marker = " <- " + target_word1
                    elif word == target_word2:
                        marker = " <- " + target_word2
                    print(f"{idx:2d}. {word}: {score:.4f}{marker}")
                
                found_solution = True
                break
            else:
                # Выводим только близкие комбинации
                if pos1 <= 15 and pos2 <= 15:
                    print(f"{word1} + {word2}: концерт({pos1}), балетмейстер({pos2})")
                
        if found_solution:
            break
    
    # Используем вычитание 
    if not found_solution:
        print("\n" + "="*50)
        print("Способ: комбинации с вычитанием")
        print("="*50)
        
        # Слова которые могут быть "лишними" в контексте
        negative_words = ["кино_NOUN", "фильм_NOUN", "книга_NOUN", "улица_NOUN", "дом_NOUN"]
        negative_candidates = [w for w in negative_words if w in model.key_to_index]
        
        for word1 in concert_candidates[:10]:
            for word2 in balletmaster_candidates[:10]:
                for neg_word in negative_candidates[:3]:
                    if word1 == word2:
                        continue
                        
                    similar_words = find_similar_words(model, 
                                                      positive=[word1, word2],
                                                      negative=[neg_word],
                                                      topn=15)
                    if not similar_words:
                        continue
                        
                    words_list = [word for word, score in similar_words]
                    
                    pos1 = words_list.index(target_word1) + 1 if target_word1 in words_list else 0
                    pos2 = words_list.index(target_word2) + 1 if target_word2 in words_list else 0
                    
                    if pos1 <= 10 and pos2 <= 10:
                        print(f"Комбинация: {word1} + {word2} - {neg_word}")
                        print(f"Позиции: {target_word1} - {pos1}, {target_word2} - {pos2}")
                        found_solution = True
                        break
                        
                if found_solution:
                    break
            if found_solution:
                break
    
    #  Тройные комбинации
    if not found_solution:
        print("\n" + "="*50)
        print("Способ: тройные комбинации")
        print("="*50)
        
        for i in range(len(concert_candidates[:8])):
            for j in range(len(balletmaster_candidates[:8])):
                for k in range(len(concert_candidates[:8])):
                    if i == k:
                        continue
                        
                    word1 = concert_candidates[i]
                    word2 = balletmaster_candidates[j]
                    word3 = concert_candidates[k]
                    
                    similar_words = find_similar_words(model, [word1, word2, word3], topn=15)
                    if not similar_words:
                        continue
                        
                    words_list = [word for word, score in similar_words]
                    
                    pos1 = words_list.index(target_word1) + 1 if target_word1 in words_list else 0
                    pos2 = words_list.index(target_word2) + 1 if target_word2 in words_list else 0
                    
                    if pos1 <= 10 and pos2 <= 10:
                        print(f"\nКомбинация найдена")
                        print(f"Комбинация: {word1} + {word2} + {word3}")
                        print(f"Позиции: {target_word1} - {pos1}, {target_word2} - {pos2}")
                        found_solution = True
                        break
                        
                if found_solution:
                    break
            if found_solution:
                break

    if not found_solution:
        print("\nНе удалось найти комбинацию для этих слов")
        print("Возможно, слова слишком далеки по смыслу в данной модели")

if __name__ == "__main__":
    main()