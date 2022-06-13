import sys
import argparse
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


def delete_empty(documents, lemmas):
    """ The functions that removes documents with empty lemmas list from document collection.
    """
    empty_ind = [i for i in range(len(lemmas_documents)) if lemmas_documents[i] == []]
    for i in reversed(empty_ind):
        documents.pop(i); lemmas.pop(i)
            

def prod(l): 
    """ The functions that computes the product of the list items. 
    """
    pr = 1
    for x in l: pr *= x
    return pr


def create_parser():
    """
    The function that creates a parser for command-line options.
    """
    pars = argparse.ArgumentParser(description="The closest documents for the given request")
    pars.add_argument('-req', '--request', default="requests/request1.txt", help="Path to request file", type=str)
    pars.add_argument('-arts', '--articles', default=["articles/article1.txt", "articles/article2.txt", "articles/article3.txt"], 
        # "article1_2.txt", "article2_2.txt", "article2_3.txt", "article2_4.txt", "article3_2.txt", "article3_3.txt"], 
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

    # print([document for i, document in enumerate(documents_collection) if lemmas_documents[i] == []])
    # ['A Social Call (2017), Prestige Records', 'Love & Liberation (2019), Concord Jazz[en]']
    # У некоторых документов получился пустой список лемм!
    delete_empty(documents_collection, lemmas_documents)

    p_request = [sum(map(lambda doc: doc.count(lemma), lemmas_documents)) / sum(map(len, lemmas_documents)) 
                for lemma in lemmas_request]
    p_documents = [[lemmas.count(lemma) / len(lemmas) for lemma in lemmas_request] 
                    for lemmas in lemmas_documents]

    # print([lemma for i, lemma in enumerate(lemmas_request) if p_request[i] == 0])
    # ['знаменитый', 'прослушать', 'танцор']
    # Некоторые слова запроса не встречаются ни в одном документе!
    p_request = [p if p != 0 else 1 / sum(map(len, lemmas_documents))  for p in p_request]
    
    # print(*zip(lemmas_request, p_request), sep='\n', end='\n\n')

    print("Request:", request)
    for Lambda in (0.5, 0.9):

        p_req_doc = [prod((1-Lambda) * p_req + Lambda * p_doc for p_req, p_doc in zip(p_request, p_documents[i])) 
                    for i in range(len(documents_collection))]

        rated_documents = sorted(zip(documents_collection, p_req_doc), key=lambda x: x[1], reverse=True)

        """
        print(f"\n10 most similar documents (lambda={Lambda}):", end='\n\n')
        print(*[f'{doc} (p(Q|d)={p})' for doc, p in rated_documents[:10]], sep='\n', end='\n\n')
        """
        print(f"\nSorted by relevance documents (lambda={Lambda}):", end='\n\n')
        for doc, p in rated_documents:
            if p == 0: break
            print(f'{doc} (p(Q|d)={p})')
        print("\n")

