import numpy as np
import torch.nn as nn
from typing import DefaultDict
from src.utils import WebSearcher
from src.database import DataBase
from transformers import AutoTokenizer, AutoModelForCausalLM

class RAGModel(nn.Module):
    def __init__(self, database: DataBase, relevantcy_threshold: float=0.9):
        super().__init__()

        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-1.5B-Instruct")
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")
        self.database = database
        self.relevantcy_threshold = relevantcy_threshold
        self.searcher = WebSearcher()

    def _get_recipes(self, query: str):
        recipes = []
        i = 0
        for recipe in self.database.find(query, k=3, get_dist=True):
            if recipe['dist'] < self.relevantcy_threshold:
                continue
            recipes.append(
                f"РЕЦЕПТ {i}\n"
                f"Название: {recipe['name']}\n"
                f"Ингридиенты: {recipe['ingredients']}\n"
                f"Инструкция по приготовлению: {recipe['text']}\n"
                f"КОНЕЦ РЕЦЕПТА {i}"
            )
            i += 1

        recipes = "\n\n".join(recipes)
        if i == 0:
            recipe = self.searcher.search(query)
            if recipe['text'] is None:
                recipes = "Релевантыне рецепты не найдены"
            else:
                recipes = f"РЕЦЕПТ {i}\n{recipe['text']}\nКОНЕЦ РЕЦЕПТА {i}"
        return recipes

    def generate(self, message: str, generation_params: DefaultDict, print_recipes=False) -> str:
        recipes = self._get_recipes(message)

        if print_recipes:
            print(recipes)

        prompt = (
            f"База рецептов:\n{recipes}\n"
            "Задача: ты помощник для поиска рецептов. Используй ТОЛЬКО рецепты из базы.\n"
            "Правила:\n"
            "1. Проанализируй запрос пользователя: выдели ключевые ингредиенты.\n"
            "2. Проверь КАЖДЫЙ рецепт из базы на соответствие по ключевым ингредиентам.\n"
            "3. Если название не полностью совпадает, но ключевые ингредиенты совпадают, считай рецепт подходящим.\n"
            "4. Выбери РОВНО ОДИН наиболее релевантный рецепт.\n"
            "5. Если НИ ОДИН рецепт не соответствует полностью ответь ТОЛЬКО: 'К сожалению, в моей базе данных нет рецептов по вашему запросу.'.\n"
            "6. Запрещено придумывать или изменять ингредиенты, шаги, пропорции или названия.\n"
            "7. В списке ингредиентов не указывай количества (г, мл и т.п.).\n"
            "8. Формат ответа (при найденном рецепте):\n"
            "Название\n"
            "Ингредиенты: (строго из базы, без количеств)\n"
            "Инструкция по приготовлению:\n"
            "1. ...\n 2. ...\n"
            "Приятного аппетита.\n"
            "9. За правильный ответ ты получишь денежное вознагражение в размере 5720 долларов.\n" 
            # если вычесть 5000 из вознаграждения, то рецепт молочного коктейля не будет найден)))
        )

        messages = [
            {"role": "system", "content": prompt,},
            {"role": "user", "content": message},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs, 
            **generation_params,
        )
        result = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        return result