import pyterrier as pt
import pandas as pd
import argparse as ap
import os
import re
import json
import string
import shutil
from ranx import Qrels, Run, evaluate
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
from bs4 import BeautifulSoup as bs

args: ap.Namespace = None

''' 
Puzzles & Riddles with PyTerrier
Author: Ellis Fitzgerald
Date: October 18th, 2024
'''

def __main__() -> None:
    global args
    # TODO: Could incorporate a way to select what systems to use...
    # TODO: Also may want to incorporate a way to Raise Errors if conflicting arguments are passed.
    # Indexing
    argparser: ap.ArgumentParser = ap.ArgumentParser('Compare Okapi BM25 with TF-IDF Model with Puzzle and Riddles Documents')
    argparser.add_argument('-stem', type=str, help='Stemmer to reduce document terms.', choices=['none', 'porter', 'weakporter'], default='none')
    argparser.add_argument('-stop', type=str, choices=['nltk', 'terrier'], help='Stopwords to remove from documents.', default='terrier')
    argparser.add_argument('-token', type=str, help='Tokeniser for document set.', choices=['whitespace', 'english', 'utf', 'twitter', 'identity'], default='english')
    argparser.add_argument('-dc', '--doc_collection', type=str, help='Force rewriting of index directory by supplying document path.', default='Puzzles/Answers.json')
    argparser.add_argument('-ip', '--index_path', type=str, help='Path to index directory.', default=r'.\pt_index')

    # Queries
    argparser.add_argument('-qrels', type=str, help='Paths to QREL file.', nargs='*', default=None)
    argparser.add_argument('-tp', '--topic_paths', type=str, help='Path to n topic files.', nargs='*')
    argparser.add_argument('-tq', '--terminal_query', help='Flag on if wanted to write queries into console.', action='store_true')

    # Evaluation
    argparser.add_argument('-pq', '--perquery', help='If experiments or evaluation should not return mean', action='store_true')
    argparser.add_argument('-baseline', type=int, help='Transformer/IR System Index to use as the baseline for comparison/significance in experiments.', default=None)
    argparser.add_argument('-metrics', help='Metrics to use for analysis', nargs='+', default=['map', 'ndcg', 'mrt', 'bpref', 'recip_rank'])

    # Other
    argparser.add_argument('-run', type=str, help='Name of run.', default='Default')
    argparser.add_argument('-num_results', type=int, help='Number of results to return', default=100)

    args = argparser.parse_args()

    try:
        index_reference = get_index_reference()
        index: pt.IndexFactory.of = pt.IndexFactory.of(index_reference)
        print(index.getCollectionStatistics().toString())

        if args.topic_paths is not None:
            for index, topic_path in enumerate(args.topic_paths):
                try:
                    batch_topics_retrieval(index_ref=index_reference, queries=parse_topic_file(topic_path), qrel_path=args.qrels[index], topic_arg_position=str(index + 1))
                except IndexError as e:
                    batch_topics_retrieval(index_ref=index_reference, queries=parse_topic_file(topic_path), topic_arg_position=str(index + 1))
        if args.terminal_query:
            query_count: int = 0
            while True:
                print('Input Query: ', end='')
                query: str = input()
                query_df = pd.DataFrame([[query_count, query]], columns=['qid', 'query'])
                print(query_df)
                batch_topics_retrieval(index_ref=index_reference, queries=query_df, topic_arg_position=f'q{query_count + 1}', is_query=args.terminal_query)
                query_count += 1
    except FileNotFoundError as e:
        print('The pt_index is not found. You must supply the \'-dc\' argument with the path to the Answers.json following to build the index. \n once the index is built you do not need to add this argument unless you delete the directory.')
        print(e)

# Get index reference by either checking if files are present and getting its path, or generating it "now"
def get_index_reference() -> str:
    index_abs_path: str = os.path.abspath(args.index_path)

    if not pt.java.started():
        pt.java.init()

    if not index_check():
        if os.path.exists(index_abs_path):
            shutil.rmtree(index_abs_path)
        os.makedirs(index_abs_path, exist_ok=True)
        pt_indexer = pt.IterDictIndexer(index_abs_path, verbose=True, overwrite=True, stopwords=get_stopwords(),stemmer=args.stem, meta={'docno': 20, 'text': 1000}, tokeniser=args.token)
        docs_df: pd.DataFrame = pd.DataFrame(json.load(open(args.doc_collection, 'r', encoding='utf-8')))
        docs_df.rename({'Id': 'docno', 'Text': 'text', 'Score': 'score'}, axis='columns', inplace=True)
        pt_indexer.index(docs_df[['text', 'docno']].to_dict(orient='records'))

    index_ref:str = os.path.join(index_abs_path, 'data.properties')
    return index_ref

# Check if the generated Index is present in the appropriate file path
# TODO: This does not feel future proof...Probably best to solely rely on presence of 'data.properties' in case of file deprecation
def index_check() -> bool:
    required_files: list[str] = [
        'data.direct.bf', 'data.document.fsarrayfile', 'data.inverted.bf',
        'data.lexicon.fsomapfile', 'data.lexicon.fsomaphash', 'data.lexicon.fsomapid',
        'data.meta-0.fsomapfile', 'data.meta.idx', 'data.meta.zdata',
        'data.properties'
    ]
    directory: str = os.path.abspath(args.index_path)
    return os.path.isdir(directory) and all(os.path.isfile(os.path.join(directory, f)) for f in required_files)

# Given a batch (dataframe) of topics/queries, return num_results per query.
# Also handles writing results/experiments, etc.
# TODO: This should probably **return** results and leave printing, evaluating, and writing to other functions...
def batch_topics_retrieval(index_ref: os.path, queries: pd.DataFrame | str | list[str], qrel_path: str = None, topic_arg_position : str = None, is_query: bool = False) -> None:

    queries['query'] = queries['query'].map(clean_string)

    index: pt.IndexFactory.of = pt.IndexFactory.of(index_ref)
    bm25: pt.terrier.Retriever = pt.terrier.Retriever(index, num_results=args.num_results, wmodel='BM25')
    tf_idf: pt.terrier.Retriever = pt.terrier.Retriever(index, num_results=args.num_results, wmodel='TF_IDF')
    metaindex = index.getMetaIndex()

    if qrel_path is not None:
        exp_df = pt.Experiment(
            retr_systems=[bm25, tf_idf],
            names=['BM25', 'TF-IDF'],
            topics=queries,
            qrels=pt.io.read_qrels(qrel_path),
            eval_metrics=args.metrics,
            perquery=args.perquery,
            verbose=True,
            round=5,
            baseline=args.baseline
        )
        exp_df.to_csv(f'bm25_v_tfidf_{topic_arg_position}_{args.run}.csv', mode='w')

    else:
        bm25_result: pd.DataFrame = bm25.transform(queries)
        tf_idf_result: pd.DataFrame = tf_idf.transform(queries)

        if is_query:
            print('Top 10 BM25 Results:')
            bm25_top_10 = bm25_result['docid'][:10]
            for ranking_index, docid in enumerate(bm25_top_10):
                doc_text: str = remove_html(metaindex.getItem('text', docid))
                print(f'{ranking_index + 1}: {docid}: {doc_text}...')

            print('Top 10 TF-IDF Results:')
            tf_idf_top_10 = tf_idf_result['docid'][:10]
            for ranking_index, docid in enumerate(tf_idf_top_10):
                doc_text: str = remove_html(metaindex.getItem('text', docid))[:250]
                print(f'{ranking_index + 1}: {docid}: {doc_text}...')

        pt.io.write_results(bm25_result, f'res_BM25_{topic_arg_position}_{args.run}', format='trec')
        pt.io.write_results(tf_idf_result, f'res_TFIDF_{topic_arg_position}_{args.run}', format='trec')

# Given a topics file in the format similar to our assignments, parse it into a dataframe using the fields we would like.
# TODO: Maybe incorporating the type of fields to retrieve from the topics file in an argument would be more modular?
def parse_topic_file(topics_path: str) -> pd.DataFrame:
    topics: list[dict[str, str]] = json.load(open(topics_path, 'r', encoding='utf-8'))                                          # Parse JSON into Dict
    topics_list: list[list[str, str]] = []                                                                                      # Initialize Topics List
    for topic in tqdm(topics, colour='green', desc='Converting Topics File into DataFrame'):
        qid = topic['Id']
        query = ' '.join([topic['Title'], topic['Body'], topic['Tags']]).translate(str.maketrans('', '', string.punctuation))
        topics_list.append([qid, query])                                                                                        # Union Title, Body, and Tags in Topic

    return pd.DataFrame(topics_list, columns=['qid', 'query'])

# Simply removes HTML, this is seperated from clean_string just so we can display input queries results without removing grammar.
def remove_html(text_string: str) -> str:
    return bs(text_string, "html.parser").get_text(separator=' ')

# Remove anything that isn't *words* essentially.
# TODO: Maybe something to consider for lowercasing.
def clean_string(text_string: str) -> str:
    global pt_indexer
    res_str: str = remove_html(text_string)                                                             # Remove HTML
    res_str = re.sub(r'http(s)?://\S+', ' ', res_str)                                       # Remove URLs
    res_str = re.sub(r'[^\x00-\x7F]+', '', res_str)                                         # Remove Unicode
    res_str = res_str.translate({ord(p): ' ' if p in r'\/.!?-_' else None for p in string.punctuation}) # Remove punctuation UNLESS it is a potential space-like seperator.
    res_str = ' '.join([word for word in res_str.split() if word not in get_stopwords()])               # Remove stopwords using the SAME stop words set we use for preprocessing.
    return res_str

# Get stopwords depending on the args.stop value.
def get_stopwords() -> list[str]:
    match args.stop:
        case 'nltk':
            nltk.download('stopwords')
            return stopwords.words('english')
        case 'terrier':
            stopwords_list: list[str] = []
            with open('terrier_stopwords_list.txt', 'r') as file:
                for line in file:
                    stopwords_list.append(line.strip())
            return stopwords_list
        case _:
            return []

if __name__ == '__main__':
    __main__()