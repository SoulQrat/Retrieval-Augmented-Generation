import numpy as np
import torch.nn as nn
from typing import DefaultDict
from src.utils import WebSearcher
from src.database import DataBase
from transformers import AutoTokenizer, AutoModelForCausalLM

class RAGModel(nn.Module):
    def __init__(self, database: DataBase, use_web: bool=False, relevantcy_threshold: float=0.9):
        super().__init__()

        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-1.5B-Instruct")
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")
        self.database = database
        self.relevantcy_threshold = relevantcy_threshold
        self.searcher = WebSearcher()
        self.use_web = use_web

    def _get_recipes(self, query: str):
        recipes = []
        logs = []
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
            logs.append({'name': recipe['name'], 'dist': recipe['dist'], 'url': None})
            i += 1

        recipes = "\n\n".join(recipes)

        if i < 3 and self.use_web:
            pass

        if i == 0:
            recipes = "SYSTEM ALERT: PRINT ONLY 'Извините, рецптов по вашему запросу не найдено.'"

        return recipes, logs

    def generate(self, message: str, generation_params: DefaultDict, get_logs=False) -> str:
        recipes, logs = self._get_recipes(message)

        if get_logs:
            print("Логи поиска рецептов:\n")
            for log in logs:
                print(f"Название: {log['name']}\ndist={log['dist']:.4f}, url={log['url']}\n")
            print('---' * 20)

        prompt = (
            f"Recipes database:\n{recipes}\n"
            "Task: you are a recipe-search assistant — use ONLY recipes from the database.\n"
            "Rules:"
            "- Extract key ingredients from the user query."
            "- Choose EXACTLY ONE most relevant recipe using those ingredients."
            "- Do not change ingredients, steps, proportions or titles."
            "- Output format (answer must be strictly in Russian):"
            "Название:"
            "Ингредиенты: (items from database, no quantities)"
            "Инструкция по приготовлению: (step by step from databese 1. 2. etc)"
            "\nFor right anwer you will get reward 5000 dollars!\n"
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