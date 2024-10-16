import pyterrier as pt
import pandas as pd
import argparse as ap
import os
import json
import shutil
import nltk
from nltk.corpus import stopwords

# Making a change AKA making stuff
def __main__() -> None:
    index_preprocessing()

def index_preprocessing() -> None:
    argparser: ap.ArgumentParser = ap.ArgumentParser('TLDR')
    argparser.add_argument('-stem', type=str, help='Stemmer to choose from')
    argparser.add_argument('-stop', type=str, choices=['nltk', 'terrier'], help='Stopwords to choose from: nltk, terrier')
    args: argparser.parse_args = argparser.parse_args()

    stopwords_arg: list[str] | str | None = None
    if args.stop == 'terrier':
        stopwords_arg = args.stop
    elif args.stop == 'nltk':
        nltk.download('stopwords')
        stopwords_arg = stopwords.words('english')

    index_rel_path: str = r'.\pt_index'
    index_abs_path: str = os.path.abspath(index_rel_path)

    if not pt.java.started():
        pt.java.init()

    if os.path.exists(index_abs_path):
        shutil.rmtree(index_abs_path)
    os.makedirs(index_abs_path, exist_ok=True)

    docs_df: pd.DataFrame = pd.DataFrame(json.load(open('Puzzles/Answers.json', 'r', encoding='utf-8')))
    docs_df.rename({'Id': 'docno', 'Text': 'text', 'Score': 'score'}, axis='columns', inplace=True)

    pd_indexer: pt.IterDictIndexer = pt.IterDictIndexer(index_abs_path, verbose=True, overwrite=True, stopwords=stopwords_arg, stemmer=args.stem)
    pd_indexer.index(docs_df[['text', 'docno']].to_dict(orient='records'))

def retrieval() -> None:
    index: pt.IndexFactory.of = pt.IndexFactory.of(r'.\pt_index\data.properties')
    print(index.getCollectionStatistics().toString())


if __name__ == '__main__':
    __main__()