import json
import chromadb
from chromadb.utils import embedding_functions
from llama_cpp import Llama

class ModelHandler:
    def __init__(self):
        # Инициализация ChromaDB
        self.client = chromadb.PersistentClient(path="chroma_db")
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            "paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.collection = self.client.get_or_create_collection("computer_faq")
        
        # Инициализация Llama3
        self.llm = Llama(
            model_path="models/Llama3-Instruct-8B-RSPO.Q4_K_M.gguf",
            n_ctx=4096,
            n_threads=6,
            n_gpu_layers=50
        )
    
    def get_answer(self, question):
        # Поиск релевантного контекста
        results = self.collection.query(
            query_texts=[question],
            n_results=5,
            include=["documents", "metadatas"]
        )
        
        # Формирование динамического контекста
        context = "Возможные решения из базы знаний:\n"
        for meta, doc in zip(results["metadatas"][0], results["documents"][0]):
            context += f"- {meta['question']}: {doc}\n"
        
        prompt = f"""Ты опытный IT-специалист. Отвечай на вопросы, используя:
1. Контекст из базы знаний (если релевантен)
2. Собственные экспертные знания
3. Логические выводы

Контекст:
{context}

Вопрос: {question}

Структура ответа:
1. Основное решение (1-2 предложения)
2. Дополнительные проверки (если нужно)
3. Альтернативные варианты (если применимо)

Ответ:"""
        
        output = self.llm(prompt, max_tokens=500, temperature=0.3)
        answer = output["choices"][0]["text"].strip()

        if not answer.count('\n') >= 3:
            continuation = self.llm(
                f"Заверши ответ:\n{answer}",
                max_tokens=200,
                temperature=0.1
            )
            answer += "\n" + continuation["choices"][0]["text"].strip()

        return answer