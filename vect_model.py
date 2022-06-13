import sys
import argparse
from math import log10, sqrt
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer


def sentences_collection(file_name):
    """
    The functions that returns a collection of sentences(=documents) from a given file.
    """
    with open(file_name, 'r', encoding='utf-8') as in_file:
        sents = sent_tokenize(in_file.read(), language='russian')
        sentences = []
        for sent in sents:
            sentences += [s for s in sent.split('\n') if s != '']
    return sentences


def words_collection(document):
    """
    The functions that returns a collection of allowed words from a given sentence(=document).
    """
    stop_words = set(stopwords.words('russian')) | set(['это', 'хотя', 'пока', 'мимо', 'всё', 'нему', 'нею', 'ею', 'весь'])
    rus_chars = set(chr(i) for i in range(ord('а'), ord('а') + 32)) | set('ё')
    # forbids = set(chr(i) for i in range(ord('a'), ord('a') + 26)) | set(str(i) for i in range(10))
    marks = set(['.', ',', ':', ';', '!', '?', '...', '"', "'", '«', '»', '—', '…', '(', ')', '[', ']', '&', '$', '%', '*', '/'])
    words = word_tokenize(document, 'russian')
    words = [word[:-1] if (word[-1] in marks) else word  for word in words]
    words = [word for word in words if not word.lower() in stop_words 
            and len(word) > 2 and not set(word.lower()) - rus_chars]
    return words


def create_parser():
    """
    The function that creates a parser for command-line options.
    """
    pars = argparse.ArgumentParser(description="The closest documents for the given request")
    pars.add_argument('-req', '--request', default="requests/request1.txt", help="Path to request file", type=str)
    pars.add_argument('-arts', '--articles', default=["articles/article1.txt", "articles/article2.txt", "articles/article3.txt"], 
        #"article1_2.txt", "article2_2.txt", "article2_3.txt", "article2_4.txt", "article3_2.txt", "article3_3.txt"], 
            help="List of paths to article files", nargs='+', type=str)
    return pars



if __name__ == '__main__':
    parser = create_parser()
    params = parser.parse_args(sys.argv[1:])
    request, articles = params.request, params.articles
    with open(request, 'r', encoding='utf-8') as req: 
        request = req.read().strip()

    documents_collection = []
    for article in articles:
        documents_collection += sentences_collection(article) 
    # print(*documents_collection, sep='\n', file=open('documents.txt', 'w'))

    pm2 = MorphAnalyzer()
    lemmas_documents = [[pm2.parse(word)[0].normal_form for word in words_collection(document)] 
            for document in documents_collection]
    lemmas_request = [pm2.parse(word)[0].normal_form for word in words_collection(request)]

    lemmas_collection = set(lemma for lemmas in lemmas_documents for lemma in lemmas) \
                        | set(lemma for lemma in lemmas_request)
    # print(*lemmas_collection, sep='\n', file=open('lemmas.txt', 'w'))

    N = len(documents_collection)
    L = len(lemmas_collection)

    lemmas_df = [sum((lemma in lemmas) for lemmas in lemmas_documents) for lemma in lemmas_collection]
    # print([lemma for i, lemma in enumerate(lemmas_collection) if lemmas_df[i] == 0])
    # ['знаменитый', 'прослушать', 'танцор']
    # Некоторые слова не встречаются ни в одном документе!
    lemmas_df = [1 if df == 0 else df  for df in lemmas_df]
    lemmas_idf = [log10(N / df) for df in lemmas_df]

    for i, tf_func in enumerate([lambda doc, lem: doc.count(lem),
                                lambda doc, lem: log10(1 + doc.count(lem))]):

        documents_tf = [[tf_func(lemmas, lemma) for lemma in lemmas_collection] for lemmas in lemmas_documents]
        documents_tf_idf = [[documents_tf[i][j] * lemmas_idf[j] for j in range(L)] for i in range(N)]

        request_tf = [tf_func(lemmas_request, lemma) for lemma in lemmas_collection]
        request_tf_idf = [request_tf[j] * lemmas_idf[j] for j in range(L)]

        norms = [sqrt(sum(map(lambda x: x**2, doc_tf_idf))) for doc_tf_idf in documents_tf_idf]
        # print([document for i, document in enumerate(documents_collection) if lemmas_documents[i] == []])
        # ['A Social Call (2017), Prestige Records', 'Love & Liberation (2019), Concord Jazz[en]']
        # У некоторых документов получился пустой список лемм!
        norms = [1 if norm == 0 else norm  for norm in norms]
        documents_vectors = [[tf_idf / norm for tf_idf in doc_tf_idf] for doc_tf_idf, norm in zip(documents_tf_idf, norms)]

        norm = sqrt(sum(map(lambda x: x**2, request_tf_idf)))
        request_vector = [tf_idf / norm for tf_idf in request_tf_idf]

        weights = [sum(r * d for r, d in zip(request_vector, document_vector)) for document_vector in documents_vectors]
        rated_documents = sorted(zip(documents_collection, weights), key=lambda x: x[1], reverse=True)

        """
        if i == 0:
            print("Request:", request, end='\n\n')
            print("10 most similar documents (tf=count):", end='\n\n')
            print(*[f'{doc} (weight={w:.3f})' for doc, w in rated_documents[:10]], sep='\n', end='\n\n')
        else: 
            print("10 most similar documents (tf=log10(1+count)):", end='\n\n')
            print(*[f'{doc} (weight={w:.3f})' for doc, w in rated_documents[:10]], sep='\n')
        """

        if i == 0:
            print("Request:", request, end='\n\n')
            print("Sorted by relevance documents (tf=count):", end='\n\n')
            for doc, w in rated_documents:
                if w == 0: break
                print(f'{doc} (weight={w:.3f})')
        else: 
            print("\nSorted by relevance documents (tf=log10(1+count)):", end='\n\n')
            for doc, w in rated_documents:
                if w == 0: break
                print(f'{doc} (weight={w:.3f})')

