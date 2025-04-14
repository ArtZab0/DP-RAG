import os
import json
from typing import List, Dict, Tuple
from openai import OpenAI
import torch
from sentence_transformers import util

from rag_backend import (
    documents, all_chunks, embedding_model, device
)

def generate_test_questions(document_texts: List[str], num_questions: int = 5) -> List[str]:
    """Generate test questions from the provided documents using OpenRouter API."""
    OPENROUTER_API_KEY = os.environ['OPENROUTER_API_KEY']
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY
    )
    
    # Combine documents and create prompt
    combined_text = "\n\n".join(document_texts)
    prompt = f"""Given the following documents, generate {num_questions} diverse questions that can be answered using the information in these documents. The questions should test different aspects of understanding and require specific details from the documents to answer correctly.

Documents:
{combined_text}

Generate {num_questions} questions, focusing on key information and specific details from the documents."""

    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": prompt}]
    }]

    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout",
        messages=messages,
        temperature=0
    )
    
    # Parse response into list of questions
    response = completion.choices[0].message.content
    questions = [q.strip() for q in response.split("\n") if q.strip() and "?" in q]
    return questions[:num_questions]

def run_rag_query(query2: str, use_priorities: bool = True) -> Tuple[str, List[Dict]]:
    """Run a RAG query with or without document priorities."""
    if not all_chunks:
        return "No documents available.", []
    
    # Convert embeddings to tensors and move to correct device
    embeddings = torch.stack([torch.tensor(chunk["embedding"], device=device) for chunk in all_chunks])
    query_embedding = embedding_model.encode(query2, convert_to_tensor=True).to(device)
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    
    if use_priorities:
        # Apply priority bonus (same as in app.py)
        beta = 0.02
        bonus = torch.tensor([beta * chunk["priority"] for chunk in all_chunks], device=device)
        scores = dot_scores * (1 + bonus)
    else:
        scores = dot_scores
    
    topk = torch.topk(scores, k=min(5, len(all_chunks)))
    selected_chunks = [all_chunks[i] for i in topk.indices]
    chunk_texts = [chunk["chunk_text"] for chunk in selected_chunks]
    
    # Query the model using existing chat.py functionality
    from chat import query
    response = query(
        user_query=query2,
        documents_text=chunk_texts,
        documents_priorities=[chunk["priority"] for chunk in selected_chunks],
        b64_image_urls=[]
    )
    
    return response, selected_chunks

def evaluate_responses(question: str, dp_rag_response: str, standard_rag_response: str) -> Dict:
    """Use LLM to evaluate and compare responses."""
    OPENROUTER_API_KEY = os.environ['OPENROUTER_API_KEY']
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY
    )
    
    evaluation_prompt = f"""You are an expert judge evaluating two different RAG (Retrieval Augmented Generation) systems. Please evaluate the following responses to a question based on relevance, accuracy, and comprehensiveness.

Question: {question}

System A (DP-RAG) Response:
{dp_rag_response}

System B (Standard RAG) Response:
{standard_rag_response}

Please evaluate both responses on a scale of 1-10 for each criterion:
1. Relevance: How well does the response address the specific question asked?
2. Accuracy: How accurate and factual is the response based on the information provided?
3. Comprehensiveness: How complete and thorough is the response?

Provide your evaluation in the following format:
DP-RAG Scores:
- Relevance: [score]
- Accuracy: [score]
- Comprehensiveness: [score]

Standard RAG Scores:
- Relevance: [score]
- Accuracy: [score]
- Comprehensiveness: [score]

Explanation: [Provide a brief explanation of your scoring, highlighting key differences between the two responses]"""

    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": evaluation_prompt}]
    }]

    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout",
        messages=messages,
        temperature=0
    )
    
    return {
        "question": question,
        "dp_rag_response": dp_rag_response,
        "standard_rag_response": standard_rag_response,
        "evaluation": completion.choices[0].message.content
    }

def run_evaluation(num_questions: int = 5) -> List[Dict]:
    """Run a complete evaluation comparing DP-RAG vs standard RAG."""
    # Get document texts for question generation
    doc_texts = [chunk["chunk_text"] for chunk in all_chunks]
    
    # Generate test questions
    questions = generate_test_questions(doc_texts, num_questions)
    
    results = []
    for question in questions:
        # Run both RAG variants
        dp_rag_response, _ = run_rag_query(question, use_priorities=True)
        standard_rag_response, _ = run_rag_query(question, use_priorities=False)
        
        # Get evaluation
        evaluation = evaluate_responses(question, dp_rag_response, standard_rag_response)
        results.append(evaluation)
    
    # Save results
    save_results(results)
    return results

def save_results(results: List[Dict]):
    """Save evaluation results to a JSON file."""
    if not os.path.exists('evaluation_results'):
        os.makedirs('evaluation_results')
    
    # Generate timestamp for filename
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    filepath = f'evaluation_results/eval_{timestamp}.json'
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

def calculate_summary_metrics(results: List[Dict]) -> Dict:
    """Calculate summary metrics from evaluation results."""
    if not results:
        return {}
    
    dp_rag_scores = {"relevance": [], "accuracy": [], "comprehensiveness": []}
    std_rag_scores = {"relevance": [], "accuracy": [], "comprehensiveness": []}
    wins = {"dp_rag": 0, "std_rag": 0, "tie": 0}
    
    for result in results:
        eval_text = result["evaluation"]
        
        # Parse scores from evaluation text
        dp_section = eval_text[eval_text.find("DP-RAG Scores:"):eval_text.find("Standard RAG Scores:")]
        std_section = eval_text[eval_text.find("Standard RAG Scores:"):eval_text.find("Explanation:")]
        
        # Extract scores using regex
        import re
        
        def extract_scores(section: str) -> Dict[str, float]:
            scores = {}
            metrics = ["Relevance", "Accuracy", "Comprehensiveness"]
            for metric in metrics:
                match = re.search(f"{metric}: (\d+)", section)
                if match:
                    scores[metric.lower()] = float(match.group(1))
            return scores
        
        dp_scores = extract_scores(dp_section)
        std_scores = extract_scores(std_section)
        
        # Add scores to lists
        for metric in dp_rag_scores:
            if metric in dp_scores:
                dp_rag_scores[metric].append(dp_scores[metric])
            if metric in std_scores:
                std_rag_scores[metric].append(std_scores[metric])
        
        # Calculate winner for this question
        dp_avg = sum(dp_scores.values()) / len(dp_scores)
        std_avg = sum(std_scores.values()) / len(std_scores)
        if dp_avg > std_avg:
            wins["dp_rag"] += 1
        elif std_avg > dp_avg:
            wins["std_rag"] += 1
        else:
            wins["tie"] += 1
    
    # Calculate averages
    def calc_averages(scores: Dict[str, List[float]]) -> Dict[str, float]:
        return {metric: sum(values) / len(values) if values else 0 
                for metric, values in scores.items()}
    
    dp_averages = calc_averages(dp_rag_scores)
    std_averages = calc_averages(std_rag_scores)
    
    # Calculate overall scores
    dp_overall = sum(dp_averages.values()) / len(dp_averages)
    std_overall = sum(std_averages.values()) / len(std_averages)
    
    # Calculate win rates
    total_questions = len(results)
    win_rates = {
        "dp_rag": (wins["dp_rag"] / total_questions) * 100,
        "std_rag": (wins["std_rag"] / total_questions) * 100,
        "tie": (wins["tie"] / total_questions) * 100
    }
    
    # Calculate improvement percentages
    improvements = {
        metric: ((dp_averages[metric] - std_averages[metric]) / std_averages[metric] * 100)
        for metric in dp_averages
    }
    improvements["overall"] = ((dp_overall - std_overall) / std_overall * 100)
    
    return {
        "dp_rag": {
            "scores": dp_averages,
            "overall": dp_overall
        },
        "std_rag": {
            "scores": std_averages,
            "overall": std_overall
        },
        "win_rates": win_rates,
        "improvements": improvements,
        "total_questions": total_questions
    }

def load_latest_results() -> List[Dict]:
    """Load the most recent evaluation results."""
    if not os.path.exists('evaluation_results'):
        return []
    
    files = os.listdir('evaluation_results')
    if not files:
        return []
    
    latest_file = max(files)
    with open(f'evaluation_results/{latest_file}', 'r') as f:
        return json.load(f)
