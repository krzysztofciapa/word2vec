import os
import argparse
from src.evaluate import Evaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate Word2Vec embeddings")
    parser.add_argument("--weights", type=str, default="model_weights.npz",
                        help="Path to model weights file")
    parser.add_argument("--analogies", action="store_true",
                        help="Run word analogy benchmark")
    parser.add_argument("--wordsim", action="store_true",
                        help="Run WordSim-353 benchmark")
    parser.add_argument("--tsne", action="store_true",
                        help="Plot t-SNE visualization")
    parser.add_argument("--all", action="store_true",
                        help="Run all evaluations")
    parser.add_argument("--save-json", type=str, default="",
                        help="Path to save evaluation metrics as JSON")
    args = parser.parse_args()

    # default setting in case no flags were added
    if not any([args.analogies, args.wordsim, args.tsne, args.all]):
        args.analogies = True
        args.wordsim = True

    evaluator = Evaluator(args.weights)
    all_results = {}

    # nearest neighbors - sanity check
    test_words = ["king", "france", "computer", "water", "one", "good"]
    print("\n--- Nearest Neighbors (cosine similarity) ---")
    for word in test_words:
        result = evaluator.get_similar_words(word, k=5)
        if isinstance(result, str):
            print(f"  {word}: {result}")
        else:
            neighbors = ", ".join(f"{w} ({s:.3f})" for w, s in result)
            print(f"  {word:>10s} → {neighbors}")
    print()

    # word analogy benchmark
    if args.analogies or args.all:
        analogy_path = os.path.join("data", "questions-words.txt")
        res = evaluator.evaluate_analogies(analogy_path)
        if res:
            all_results['analogy_accuracy'] = res['overall_accuracy']

    #wordsim353
    if args.wordsim or args.all:
        wordsim_path = os.path.join("data", "wordsim353.tsv")
        res = evaluator.evaluate_word_similarity(wordsim_path)
        if res is not None:
            all_results['wordsim_spearman'] = res

    #save to JSON
    if args.save_json:
        import json
        with open(args.save_json, 'w') as f:
            json.dump(all_results, f, indent=4)
        print(f"\nSaved evaluation metrics to {args.save_json}")

    # tSNE visualisation
    if args.tsne or args.all:
        evaluator.plot_embeddings(num_words=400, method='tsne')


if __name__ == "__main__":
    main()