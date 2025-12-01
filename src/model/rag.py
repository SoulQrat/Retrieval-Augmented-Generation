import numpy as np
import torch.nn as nn
from typing import DefaultDict
from src.database import DataBase
from transformers import AutoTokenizer, AutoModelForCausalLM

class RAGModel(nn.Module):
    def __init__(self, database: DataBase):
        super().__init__()

        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-1.5B-Instruct")
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")
        self.database = database

    def generate(self, message: str, generation_params: DefaultDict) -> str:
        recipes = []
        for i, recipe in enumerate(self.database.find(message, k=3)):
            recipes.append(
                f"РЕЦЕПТ {i}\n"
                f"Название: {recipe['name']}\n"
                f"Ингридиенты: {recipe['ingredients']}\n"
                f"Инструкция по приготовлению: {recipe['text']}\n"
                f"КОНЕЦ РЕЦЕПТА {i}"
            )

        recipes = "\n\n".join(recipes)
        if len(recipes) == 0:
            recipes = "Релевантыне рецепты не найдены"

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