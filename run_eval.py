from src.evaluate import Evaluator

def main():

    evaluator = Evaluator("model_weights.npz")
    

    test_words = ["king", "france", "computer", "water", "one"]
    
    for word in test_words:
        print(f"\nClosest words to '{word}':")
        results = evaluator.get_similar_words(word, k=5)
        if isinstance(results, str):
            print(results)
        else:
            for neighbor, sim in results:
                print(f" -> {neighbor} (cosine prob: {sim:.3f})")


    evaluator.plot_embeddings(num_words=400, method='tsne')

if __name__ == "__main__":
    main()