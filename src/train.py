import os
import argparse
import numpy as np
import time

from data_loader import DataLoader, TextStreamer, Vocabulary
from model import Model
from optim import SGD

def train(text_stream, vocab, args):

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    WINDOW_SIZE = args.window_size
    NEG_SAMPLES = args.neg_samples
    EMBEDDING_DIM = args.embedding_dim
    LEARNING_RATE = args.lr

    print("-" * 50)
    print(f"Training started. Vocab size: {len(vocab.word2id)} tokens.")
    print(f"Dimensions: {EMBEDDING_DIM}, Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}")
    print("-" * 50)


    model = Model(vocab_size=len(vocab.word2id), embedding_dim=EMBEDDING_DIM)
    
    if hasattr(vocab, 'total_words') and vocab.total_words > 0:
        total_batches_est = (vocab.total_words * EPOCHS * (2 * WINDOW_SIZE)) // BATCH_SIZE
    else:
        total_batches_est = 100000 
        
    optimizer = SGD(model, lr=LEARNING_RATE, total_steps=total_batches_est)
    
    for epoch in range(EPOCHS):
        
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
        tokens_per_sec = 0.0
        
        for centers, contexts, negatives in dataloader:
            
            batch_loss = model.train_step(centers, contexts, negatives, optimizer)
            
            total_loss += batch_loss
            batches_processed += 1
            total_tokens += len(centers)
            
            if batches_processed % 1000 == 0:
                epoch_elapsed = time.time() - start_time
                if epoch_elapsed > 0:
                    tokens_per_sec = total_tokens / epoch_elapsed
                    print(f"Epoch {epoch+1} | Batch {batches_processed} | LR: {optimizer.current_lr:.5f} | Tokens/s: {tokens_per_sec:.2f}")


        epoch_duration = time.time() - start_time
        if epoch_duration > 0:
            tokens_per_sec = total_tokens / epoch_duration
            
        final_epoch_loss = total_loss / max(1, batches_processed)
        print(f"=== Epoch {epoch+1} summary ===")
        print(f"Time: {epoch_duration:.2f} s | Final loss: {final_epoch_loss:.4f} | Tokens/s: {tokens_per_sec:.2f}")
        print("-" * 50)

        model.save_weights("model_weights.npz", vocab.word2id)

    return model

def main():
    parser = argparse.ArgumentParser(description="Trenowanie modelu Word2Vec w czystym NumPy")
    parser.add_argument("--corpus", type=str, default=os.path.join("data", "text8"), help="Ścieżka do korpusu treningowego")
    parser.add_argument("--epochs", type=int, default=1, help="Liczba epok")
    parser.add_argument("--batch-size", type=int, default=128, help="Rozmiar batcha")
    parser.add_argument("--window-size", type=int, default=2, help="Liczba słów po jednej stronie okna")
    parser.add_argument("--neg-samples", type=int, default=5, help="Liczba próbek negatywnych na słowo")
    parser.add_argument("--embedding-dim", type=int, default=100, help="Wymiar wektorów")
    parser.add_argument("--lr", type=float, default=0.025, help="Początkowy Learning Rate")
    args = parser.parse_args()

    corpus_path = args.corpus
    
    if not os.path.exists(corpus_path):
        print(f"Error: Dataset '{corpus_path}' not found. Run download_data.py first.")
        return
    streamer = TextStreamer(corpus_path)
    
    vocab = Vocabulary(min_count=5) 
    
    print("Building vocabulary...")
    vocab.build_vocabulary(streamer)

    print("\n--- Top 10 words most likely to be excluded ---")
    sorted_words = sorted(vocab.word_counts.items(), key=lambda item: item[1], reverse=True)
    
    for word_id, count in sorted_words[:10]:
        word = vocab.id2word[word_id]
        p_discard = vocab.discard_probs.get(word_id, 0.0)
        print(f"Token: '{word:<6}' | seen_count: {count:<7} | exclusion prob: {p_discard * 100:.1f}%")
    print("-" * 50 + "\n")

    
    if len(vocab) == 0:
        print("Error: Vocabulary is empty. Corpus parsing failed.")
        return
        
    trained_model = train(text_stream=streamer, vocab=vocab, args=args)
    print("Training phase completed successfully.")

if __name__ == "__main__":
    main()