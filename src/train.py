import os
import argparse
import numpy as np
import time

from data_loader import DataLoader, TextStreamer, Vocabulary
from model import Model
from optim import SGD, AdaGrad, SGLD


def create_optimizer(name, model, args, total_batches_est):
    if name == 'sgd':
        return SGD(model, lr=args.lr, total_steps=total_batches_est)
    elif name == 'adagrad':
        return AdaGrad(model, lr=args.lr)
    elif name == 'sgld':
        return SGLD(
            model, lr=args.lr, total_steps=total_batches_est,
            temperature=args.temperature,
            burn_in_epochs=args.burn_in_epochs,
            snapshot_interval=args.snapshot_interval,
            max_snapshots=args.max_snapshots,
            centroid_every=args.centroid_every,
        )
    else:
        raise ValueError(f"Unknown optimizer: {name}. Choose from: sgd, adagrad, sgld")


def run_gradient_check(model, vocab, batch_size=16, neg_samples=5):

    print("\n" + "=" * 50)
    print("GRADIENT CHECK (numerical vs analytic)")
    print("=" * 50)

    centers = np.random.randint(0, len(vocab.word2id), size=batch_size)
    contexts = np.random.randint(0, len(vocab.word2id), size=batch_size)
    negatives = np.random.randint(0, len(vocab.word2id), size=(batch_size, neg_samples))

    results = model.gradient_check(centers, contexts, negatives, num_checks=10)

    all_ok = True
    for param_name, rel_error in results.items():
        status = "OK" if rel_error < 1e-5 else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(f"  {param_name:15s} | relative error: {rel_error:.2e} [{status}]")

    if all_ok:
        print("All gradient checks PASSED.")
    else:
        print("WARNING: Some gradient checks failed!")
    print("=" * 50 + "\n")
    return all_ok


def train(text_stream, vocab, args):

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    WINDOW_SIZE = args.window_size
    NEG_SAMPLES = args.neg_samples
    EMBEDDING_DIM = args.embedding_dim

    print("-" * 50)
    print(f"Training config:")
    print(f"  Vocab size:     {len(vocab.word2id)} tokens")
    print(f"  Embedding dim:  {EMBEDDING_DIM}")
    print(f"  Epochs:         {EPOCHS}")
    print(f"  Batch size:     {BATCH_SIZE}")
    print(f"  Window size:    {WINDOW_SIZE}")
    print(f"  Neg samples:    {NEG_SAMPLES}")
    print(f"  Optimizer:      {args.optimizer}")
    print(f"  Learning rate:  {args.lr}")
    print("-" * 50)

    model = Model(vocab_size=len(vocab.word2id), embedding_dim=EMBEDDING_DIM)

    # run gradient check if requested
    if args.gradient_check:
        run_gradient_check(model, vocab, batch_size=BATCH_SIZE, neg_samples=NEG_SAMPLES)

    if hasattr(vocab, 'total_words') and vocab.total_words > 0:
        total_batches_est = (vocab.total_words * EPOCHS * (2 * WINDOW_SIZE)) // BATCH_SIZE
    else:
        total_batches_est = 100000

    optimizer = create_optimizer(args.optimizer, model, args, total_batches_est)

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

        for centers, contexts, negatives in dataloader:

            batch_loss = model.train_step(centers, contexts, negatives, optimizer, current_epoch=epoch)

            total_loss += batch_loss
            batches_processed += 1
            total_tokens += len(centers)

            if batches_processed % 1000 == 0:
                epoch_elapsed = time.time() - start_time
                if epoch_elapsed > 0:
                    tokens_per_sec = total_tokens / epoch_elapsed
                    avg_loss = total_loss / batches_processed
                    lr_info = f"LR: {optimizer.current_lr:.6f}" if hasattr(optimizer, 'current_lr') else ""
                    snap_info = f" | Snapshots: {len(optimizer.snapshots)}" if hasattr(optimizer, 'snapshots') else ""
                    print(f"Epoch {epoch+1} | Batch {batches_processed:>6d} | "
                          f"Loss: {avg_loss:.4f} | {lr_info} | "
                          f"Tokens/s: {tokens_per_sec:.0f}{snap_info}")

        epoch_duration = time.time() - start_time
        tokens_per_sec = total_tokens / max(epoch_duration, 1e-6)

        final_epoch_loss = total_loss / max(1, batches_processed)
        print(f"\n=== Epoch {epoch+1} summary ===")
        print(f"Time: {epoch_duration:.1f}s | Loss: {final_epoch_loss:.4f} | "
              f"Tokens/s: {tokens_per_sec:.0f} | Batches: {batches_processed}")
        print("-" * 50)

        # save weights (use centroid for SGLD if available)
        W_in_override = None
        snapshots = None
        if hasattr(optimizer, 'get_centroid') and optimizer.get_centroid() is not None:
            W_in_override = optimizer.get_centroid()
            print(f"  [SGLD] Using centroid embedding (from {optimizer.centroid_count} samples)")
        if hasattr(optimizer, 'get_snapshots'):
            snapshots = optimizer.get_snapshots()
        model.save_weights(args.output, vocab.word2id,
                           snapshots=snapshots, W_in_override=W_in_override)

    return model


def main():
    import tracemalloc
    tracemalloc.start()
    parser = argparse.ArgumentParser(description="Word2Vec Skip-Gram with Negative Sampling (pure NumPy)")

    # data
    parser.add_argument("--corpus", type=str, default=os.path.join("data", "text8"),
                        help="Path to training corpus")

    # model
    parser.add_argument("--embedding-dim", type=int, default=100, help="Embedding dimension")
    parser.add_argument("--window-size", type=int, default=5, help="Context window radius")
    parser.add_argument("--neg-samples", type=int, default=5, help="Number of negative samples")

    # training
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.025, help="Initial learning rate")

    # pptimizer selection
    parser.add_argument("--optimizer", type=str, default="sgd",
                        choices=["sgd", "adagrad", "sgld"],
                        help="Optimizer: sgd, adagrad, or sgld")

    # SGLD-specific
    parser.add_argument("--temperature", type=float, default=1e-4,
                        help="SGLD noise temperature τ (noise_std = √(2·lr·τ))")
    parser.add_argument("--burn-in-epochs", type=int, default=1,
                        help="SGLD burn-in epochs (pure SGD before noise starts)")
    parser.add_argument("--snapshot-interval", type=int, default=5000,
                        help="SGLD steps between weight snapshots")
    parser.add_argument("--max-snapshots", type=int, default=20,
                        help="Maximum number of SGLD snapshots to keep")
    parser.add_argument("--centroid-every", type=int, default=100,
                        help="SGLD steps between centroid (running mean) updates")

    # debug
    parser.add_argument("--gradient-check", action="store_true",
                        help="Run numerical gradient verification before training")
    parser.add_argument("--output", type=str, default="model_weights.npz",
                        help="Path to save the model weights")

    args = parser.parse_args()

    corpus_path = args.corpus

    if not os.path.exists(corpus_path):
        print(f"Error: Dataset '{corpus_path}' not found. Run download_data.py first.")
        return
    streamer = TextStreamer(corpus_path)

    vocab = Vocabulary(min_count=5)

    print("Building vocabulary...")
    vocab.build_vocabulary(streamer)

    print(f"\n--- Top 10 most frequent words (subsampling) ---")
    sorted_words = sorted(vocab.word_counts.items(), key=lambda item: item[1], reverse=True)

    for word_id, count in sorted_words[:10]:
        word = vocab.id2word[word_id]
        p_discard = vocab.discard_probs.get(word_id, 0.0)
        print(f"  '{word:<6}' | count: {count:<7} | discard prob: {p_discard * 100:.1f}%")
    print("-" * 50 + "\n")

    if len(vocab) == 0:
        print("Error: Vocabulary is empty. Corpus parsing failed.")
        return

    trained_model = train(text_stream=streamer, vocab=vocab, args=args)
    
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print("\n" + "="*50)
    print("HARDWARE BENCHMARK")
    print(f"  Peak RAM Usage: {peak_mem / 1024 / 1024:.1f} MB")
    print("="*50)
    
    print("Training completed successfully.")


if __name__ == "__main__":
    main()