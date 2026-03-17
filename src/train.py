import os

import numpy as np
import time

from data_loader import DataLoader, TextStreamer, Vocabulary
from model import Model


def train(text_stream, vocab):

    EPOCHS = 1
    BATCH_SIZE = 128
    WINDOW_SIZE = 2
    NEG_SAMPLES = 5
    EMBEDDING_DIM = 100
    LEARNING_RATE = 0.025

    print("-" * 50)
    print(f"Training started. Vocab size: {len(vocab.word2id)} tokens.")
    print(f"Dimensions: {EMBEDDING_DIM}, Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}")
    print("-" * 50)


    model = Model(vocab_size=len(vocab.word2id), embedding_dim=EMBEDDING_DIM)
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        dataloader = DataLoader(
            text_stream=text_stream, 
            vocabulary=vocab, 
            window_size=WINDOW_SIZE, 
            batch_size=BATCH_SIZE, 
            neg_samples=NEG_SAMPLES
        )
        
        start_time = time.time()
        total_loss = 0.0
        batches_processed = 0
        total_tokens = 0
        
        for centers, contexts, negatives in dataloader:
            
            batch_loss = model.train_step(centers, contexts, negatives, LEARNING_RATE)
            
            total_loss += batch_loss
            batches_processed += 1
            total_tokens += len(centers)
            
            if batches_processed % 1000 == 0:
                current_avg_loss = total_loss / batches_processed
                elapsed = time.time() - start_time
                tokens_per_sec = total_tokens / elapsed


        epoch_duration = time.time() - start_time
        final_epoch_loss = total_loss / max(1, batches_processed)
        print(f"=== Epoch {epoch+1} summary ===")
        print(f"Time: {epoch_duration:.2f} s | Final loss = : {final_epoch_loss:.4f} | Tokens/s: {tokens_per_sec}")
        print("-" * 50)

        model.save_weights("model_weights.npz", vocab.word2id)

    return model

def main():

    corpus_path = os.path.join("data", "text8")
    
    if not os.path.exists(corpus_path):
        print(f"Error: Dataset '{corpus_path}' not found. Run download_data.py first.")
        return
    streamer = TextStreamer(corpus_path, limit_lines= 10000)
    
    vocab = Vocabulary(min_count=5) 
    
    print("Building vocabulary...")
    vocab.build_vocabulary(streamer)
    
    if len(vocab) == 0:
        print("Error: Vocabulary is empty. Corpus parsing failed.")
        return
        
    trained_model = train(text_stream=streamer, vocab=vocab)
    print("Training phase completed successfully.")

if __name__ == "__main__":
    main()