from src.evaluate import Evaluator

def main():
    print("Ładowanie wektorów do pamięci...")
    # Inicjalizacja ewaluatora i automatyczna normalizacja L2
    evaluator = Evaluator("model_weights.npz")
    
    # Testy K-Nearest Neighbors dla wysoce frekwencyjnych słów
    test_words = ["king", "france", "computer", "water", "one"]
    
    for word in test_words:
        print(f"\nNajbliższe słowa dla '{word}':")
        results = evaluator.get_similar_words(word, k=5)
        if isinstance(results, str):
            print(results)
        else:
            for neighbor, sim in results:
                print(f" -> {neighbor} (podobieństwo kosinusowe: {sim:.3f})")

    # Wizualizacja skupień w rzucie 2D
    print("\nGenerowanie wizualizacji przestrzeni wektorowej...")
    evaluator.plot_embeddings(num_words=400, method='tsne')

if __name__ == "__main__":
    main()