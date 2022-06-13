from math import log10
from collections import defaultdict


def models_documents(file_path):
    with open(file_path, 'r', encoding='utf-8') as in_file:
        model_1, model_2 = '', ''
        documents_1, documents_2 = [], []

        for line in in_file.readlines()[1:]:
            if line.startswith('Sorted by relevance documents'):
                if not model_1: 
                    model_1 = line[line.find('(') + 1 : line.rfind(')')]
                else: 
                    model_2 = line[line.find('(') + 1 : line.rfind(')')]
            elif line.strip():
                if not model_2:
                    documents_1.append(line[:line.find('(')].strip())
                else:
                    documents_2.append(line[:line.find('(')].strip())

        documents = {model_1 : documents_1, model_2 : documents_2}
        return documents


def ideal_documents(file_path):
    with open(file_path, 'r', encoding='utf-8') as in_file:
        requests, rated_documents = [], []
        documents = defaultdict(int)

        for line in in_file:
            if line.startswith('Request:'):
                requests.append(line.split("Request:")[1].strip())
                if documents:
                    rated_documents.append(documents)
                documents = defaultdict(int)
            elif line.strip():
                doc, rating = line[:line.rfind('(')].strip(), int(line[line.rfind('(') + 1 : line.rfind(')')])
                documents[doc] = rating

        rated_documents.append(documents)
        return requests, rated_documents


def relevance_documents(model_documents, ideal_documents):
    return {doc : ideal_documents[doc] for doc in model_documents}


def dcg(rel_documents):
    return sum(rel / log10(r + 2) for r, rel in enumerate(rel_documents.values()))


vector_documents_paths = [f'vect_results/documents_request{num}_all.txt' for num in range(1, 4)]
language_documents_paths = [f'lang_results/documents_request{num}_all.txt' for num in range(1, 4)]

vector_documents = [models_documents(doc_path) for doc_path in vector_documents_paths]
language_documents = [models_documents(doc_path) for doc_path in language_documents_paths]

requests, ideal_documents = ideal_documents("ideal_documents.txt")

vector_documents = {request : {model : relevance_documents(docs[model], ideal) for model in docs} 
                    for request, docs, ideal in zip(requests, vector_documents, ideal_documents)}
language_documents = {request : {model : relevance_documents(docs[model], ideal) for model in docs} 
                    for request, docs, ideal in zip(requests, language_documents, ideal_documents)}
ideal_documents = {request : ideal for request, ideal in zip(requests, ideal_documents)}

ndcg_models = defaultdict(int)

for request in requests:
    print("Request:", request, end='\n\n')
    ndcg_vector_models = {model : dcg(vector_documents[request][model]) / dcg(ideal_documents[request]) 
                            for model in vector_documents[request]}
    ndcg_language_models = {model : dcg(language_documents[request][model]) / dcg(ideal_documents[request]) 
                            for model in language_documents[request]}

    print("NDCG measure for 2 vector models:")
    for model, ndcg in ndcg_vector_models.items():
        print(f"{model}: {ndcg:.3f}")
        ndcg_models[model] += ndcg

    print("\nNDCG measure for 2 language models:")
    for model, ndcg in ndcg_language_models.items():
        print(f"{model}: {ndcg:.3f}")
        ndcg_models[model] += ndcg
    print('\n')

print("-"*40 + "\nMean NDCG for each model (2 vector models and 2 language models):")
for model, ndcg in ndcg_models.items():
    print(f"{model}: {ndcg / len(requests):.3f}")
