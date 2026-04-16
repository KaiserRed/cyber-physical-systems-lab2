import json
import csv
import requests
import sys

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:0.5b"

def load_queries(file_path: str) -> list[str]:
    """Загружает список запросов из JSON-файла."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_response(prompt: str) -> str:
    """Отправляет запрос на сервер Ollama и возвращает ответ модели."""
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        return f"[Ошибка: {e}]"

def save_report(results: list[dict], file_path: str) -> None:
    """Сохраняет пары 'запрос-ответ' в CSV-файл с двумя колонками."""
    with open(file_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Запрос к LLM", "Вывод LLM"])
        writer.writeheader()
        for r in results:
            writer.writerow({"Запрос к LLM": r["query"], "Вывод LLM": r["response"]})

def main() -> None:
    """Основная функция: выполняет инференс и создает отчет."""
    if len(sys.argv) < 2:
        print("Использование: python script.py <путь_к_json>")
        sys.exit(1)
        
    queries = load_queries(sys.argv[1])
    results = []
    
    for i, q in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] {q[:50]}...")
        results.append({"query": q, "response": generate_response(q)})
    
    save_report(results, "inference_report.csv")
    print("Отчет сохранен в inference_report.csv")

if __name__ == "__main__":
    main()