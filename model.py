import json
import chromadb
from chromadb.utils import embedding_functions
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer, util

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

        self.semantic_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    def is_semantically_similar(self, question, answer, threshold=0.6):
        q_embed = self.semantic_model.encode(question, convert_to_tensor=True)
        a_embed = self.semantic_model.encode(answer, convert_to_tensor=True)
        score = util.cos_sim(q_embed, a_embed).item()
        return score >= threshold, score
    
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

Ответ:"""
        
        output = self.llm(prompt, max_tokens=500, temperature=0.3)
        answer = output["choices"][0]["text"].strip()
        
        is_ok, score = self.is_semantically_similar(question, answer)

        if not is_ok:
            retry_output = self.llm(prompt, max_tokens=500, temperature=0.5)
            new_answer = retry_output["choices"][0]["text"].strip()
            is_retry_ok, new_score = self.is_semantically_similar(question, new_answer)

            if is_retry_ok:
                answer = new_answer + f"\n\n(Семантическое совпадение улучшено до {round(new_score, 2)})"
            else:
                answer += f"\n\n(Ответ может быть не совсем релевантен. Совпадение: {round(score, 2)})"

        if not answer.count('\n') >= 3:
            continuation = self.llm(
                f"Заверши ответ:\n{answer}",
                max_tokens=200,
                temperature=0.1
            )
            answer += "\n" + continuation["choices"][0]["text"].strip()

        return answer