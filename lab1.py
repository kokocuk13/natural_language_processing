import nltk
from nltk.tokenize import word_tokenize
import pymorphy3

nltk.download('punkt')
nltk.download('punkt_tab')

def analyze_text(text):

    tokens = word_tokenize(text)

    #инициализируем анализатор
    morph = pymorphy3.MorphAnalyzer()

    #список хранения 
    valid_pairs = []

    for i in range (len(tokens)-1):
        word1 = tokens[i]
        word2 = tokens[i+1]

    #пропускаем знаки припинания
        if not word1.isalpha() or not word2.isalpha():
            continue

        # морфологический анализ обоих слов    
        parsed1 = morph.parse(word1)[0]
        parsed2 = morph.parse(word2)[0]

        #получение части речи для первого и второго слова
        pos1 = parsed1.tag.POS
        pos2 = parsed2.tag.POS

        #проверка на сущ или прил
        valid_pos = {'NOUN', 'ADJF', 'ADJS'}

        if pos1 not in valid_pos and pos2 not in valid_pos:
            continue
        # Получаем грамматические характеристики
        gender1 = parsed1.tag.gender  # Род первого и второго слов
        gender2 = parsed2.tag.gender  
        
        number1 = parsed1.tag.number  # Число первого и второго слов
        number2 = parsed2.tag.number  
        
        case1 = parsed1.tag.case     # Падеж первого и второго слов
        case2 = parsed2.tag.case     
        
        # Проверяем согласование по роду, числу и падежу
        # Учитываем, что некоторые характеристики могут быть None
        gender_match = (gender1 is not None and gender2 is not None and 
                       gender1 == gender2)
        number_match = (number1 is not None and number2 is not None and 
                       number1 == number2)
        case_match = (case1 is not None and case2 is not None and 
                     case1 == case2)
        
        # Если все три характеристики совпадают
        if gender_match and number_match and case_match:
            # Получаем леммы 
            lemma1 = parsed1.normal_form
            lemma2 = parsed2.normal_form
            
            # Добавляем пару в результат
            valid_pairs.append((lemma1, lemma2))
    
    return valid_pairs

def main():

    filename = input("введите файл: ")
    
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            text = file.read()
    except FileNotFoundError:
        print(f"Ошибка: Файл '{filename}' не найден!")
        return
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return
    
    
    pairs = analyze_text(text)
    
    
    if not pairs:
        print("Не найдено пар, удовлетворяющих условиям.")
    else:
        print("\nНайденные пары (в виде лемм):")
        print("-" * 40)
        for i, (word1, word2) in enumerate(pairs, 1):
            print(f"{i:2d}. {word1} {word2}")


if __name__ == "__main__":
     
    main()
