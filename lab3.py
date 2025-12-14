from transformers import BertTokenizer, BertForMaskedLM
import torch
from torch.nn import functional as F

name = 'bert-base-multilingual-uncased'
tokenizer = BertTokenizer.from_pretrained(name)
model = BertForMaskedLM.from_pretrained(name, return_dict=True)

texts = [
    "Современная [MASK] железная дорога использует цифровые технологии."
]

target_words = ["электронная", "железнодорожная"]

for text in texts:
    print(f"\nКонтекст: {text}")
    
    inputs = tokenizer(text, return_tensors="pt")
    mask_token_id = tokenizer.mask_token_id
    mask_index = torch.where(inputs["input_ids"][0] == mask_token_id)[0][0].item()
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    mask_logits = logits[0, mask_index]
    mask_probs = F.softmax(mask_logits, dim=-1)
    top_10 = torch.topk(mask_probs, 10)
    
    print("Топ-10 предсказаний:")
    found = []
    
    for i, idx in enumerate(top_10.indices):
        word = tokenizer.decode([idx]).strip()
        probability = top_10.values[i].item()
        print(f"{i+1}. {word} (вероятность: {probability:.4f})")
        
        
        if "электрон" in word.lower():
            found.append(("электронная", probability))
        elif "железнодорож" in word.lower():
            found.append(("железнодорожная", probability))
    
    if len(found) >= 2:
        print(f"\nОба целевых слова найдены в топ-10:")
        for word, prob in found:
            print(f"  - {word}: вероятность {prob:.4f}")
        break
    elif found:
        print(f"\nНайдены следующие целевые слова:")
        for word, prob in found:
            print(f"  - {word}: вероятность {prob:.4f}")
    else:
        print(f"\nЦелевые слова не найдены в топ-10.")

print("Проверка завершена.")