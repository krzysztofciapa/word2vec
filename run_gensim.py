import os
import time
import numpy as np
import logging
import argparse
from gensim.models import Word2Vec
from gensim.models.word2vec import Text8Corpus

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def train_and_export(model_type, sg, corpus_path):
    print(f"\n==============================================")
    print(f"Training Gensim {model_type.upper()}")
    print(f"==============================================")
    
    sentences = Text8Corpus(corpus_path)
    
    start_time = time.time()
    
    model = Word2Vec(
        sentences,
        vector_size=100,
        window=5,
        min_count=5,
        sample=1e-5,  
        sg=sg,  
        negative=5,
        epochs=5,
        workers=8,
        alpha=0.025,
        min_alpha=0.0001
    )
    
    duration = time.time() - start_time
    print(f"{model_type.upper()} training completed in {duration:.1f} seconds.")

    embeddings = model.wv.vectors
    word2id = model.wv.key_to_index
    
    out_path = f"models/model_gensim_{model_type}.npz"
    os.makedirs("models", exist_ok=True)
    np.savez_compressed(out_path, embeddings=embeddings, word2id=word2id)
    print(f"Exported to {out_path} for evaluation.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, default="data/text8")
    args = parser.parse_args()
    
    train_and_export("skipgram", sg=1, corpus_path=args.corpus)
    train_and_export("cbow", sg=0, corpus_path=args.corpus)
