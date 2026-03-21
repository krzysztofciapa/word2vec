import os
import numpy as np


class Evaluator:
    def __init__(self, filepath):

        data = np.load(filepath, allow_pickle=True)
        self.embeddings = data['embeddings']
        self.word2id = data['word2id'].item()
        self.id2word = {idx: word for word, idx in self.word2id.items()}

        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.norm_embeddings = self.embeddings / norms

    def get_similar_words(self, word, k=5):

        if word not in self.word2id:
            return f"Word '{word}' not found."

        word_id = self.word2id[word]
        query_vec = self.norm_embeddings[word_id]

        cosine_similarities = np.dot(self.norm_embeddings, query_vec)

        nearest_ids = np.argsort(cosine_similarities)[::-1][1:k + 1]

        return [(self.id2word[idx], cosine_similarities[idx]) for idx in nearest_ids]

  #Word analogy benchmark
    def evaluate_analogies(self, analogy_file, max_vocab=None):

        if not os.path.exists(analogy_file):
            print(f"Analogy file not found: {analogy_file}")
            return None

        if max_vocab is not None:
            vocab_set = set(list(self.word2id.keys())[:max_vocab])
        else:
            vocab_set = set(self.word2id.keys())

        sections = {}
        current_section = "unknown"
        total_correct = 0
        total_count = 0

        with open(analogy_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(':'):
                    current_section = line[2:]
                    if current_section not in sections:
                        sections[current_section] = {'correct': 0, 'total': 0}
                    continue

                parts = line.lower().split()
                if len(parts) != 4:
                    continue

                a, b, c, expected = parts

                # skip if any word is not in vocabulary
                if not all(w in vocab_set and w in self.word2id for w in [a, b, c, expected]):
                    continue

                # vec(b) - vec(a) + vec(c)
                id_a, id_b, id_c = self.word2id[a], self.word2id[b], self.word2id[c]
                query = self.norm_embeddings[id_b] - self.norm_embeddings[id_a] + self.norm_embeddings[id_c]

                # normalize query
                query_norm = np.linalg.norm(query)
                if query_norm > 0:
                    query = query / query_norm

                # find nearest, excluding a, b, c
                similarities = np.dot(self.norm_embeddings, query)
                exclude_ids = {id_a, id_b, id_c}
                for eid in exclude_ids:
                    similarities[eid] = -np.inf

                predicted_id = np.argmax(similarities)
                predicted_word = self.id2word[predicted_id]

                is_correct = (predicted_word == expected)
                if is_correct:
                    total_correct += 1
                    sections[current_section]['correct'] += 1

                total_count += 1
                sections[current_section]['total'] += 1

        if total_count == 0:
            print("No analogy questions could be evaluated (words not in vocab).")
            return None

        results = {
            'overall_accuracy': total_correct / total_count,
            'total_correct': total_correct,
            'total_count': total_count,
            'sections': {}
        }

        print(f"\n{'='*60}")
        print(f"WORD ANALOGY EVALUATION")
        print(f"{'='*60}")

        for section, counts in sections.items():
            if counts['total'] > 0:
                accuracy = counts['correct'] / counts['total']
                results['sections'][section] = {
                    'accuracy': accuracy,
                    'correct': counts['correct'],
                    'total': counts['total']
                }
                print(f"  {section:40s} {counts['correct']:>4d}/{counts['total']:<4d} = {accuracy:.1%}")

        print(f"{'─'*60}")
        print(f"  {'TOTAL':40s} {total_correct:>4d}/{total_count:<4d} = {total_correct/total_count:.1%}")
        print(f"{'='*60}\n")

        return results

#wordsim 353 benchmark - spearman correlation
    def evaluate_word_similarity(self, simlex_file):
        if not os.path.exists(simlex_file):
            print(f"Similarity file not found: {simlex_file}")
            return None

        human_scores = []
        model_scores = []
        skipped = 0

        with open(simlex_file, 'r', encoding='utf-8') as f:
            header = True
            for line in f:
                if header:
                    header = False
                    continue
                
                line = line.strip()
                if not line:
                    continue
                    
                if simlex_file.lower().endswith('.csv'):
                    parts = line.split(',')
                else:
                    parts = line.split('\t')
                    if len(parts) < 3:
                        parts = line.split()
                
                if len(parts) < 3:
                    continue

                w1, w2 = parts[0].lower(), parts[1].lower()
                try:
                    score = float(parts[2])
                except (ValueError, IndexError):
                    continue

                if w1 not in self.word2id or w2 not in self.word2id:
                    skipped += 1
                    continue

                id1, id2 = self.word2id[w1], self.word2id[w2]
                cos_sim = np.dot(self.norm_embeddings[id1], self.norm_embeddings[id2])

                human_scores.append(score)
                model_scores.append(cos_sim)

        if len(human_scores) < 2:
            print("Not enough word pairs found in vocabulary for similarity eval.")
            return None

        rho = _spearman_correlation(human_scores, model_scores)

        print(f"\n{'='*60}")
        print(f"WORD SIMILARITY EVALUATION")
        print(f"{'='*60}")
        print(f"  Pairs evaluated: {len(human_scores)}")
        print(f"  Pairs skipped (OOV): {skipped}")
        print(f"  Spearman ρ: {rho:.4f}")
        print(f"{'='*60}\n")

        return rho

#plots
    def plot_embeddings(self, num_words=300, method='tsne'):

        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt

        vectors_to_plot = self.norm_embeddings[:num_words]
        words_to_plot = [self.id2word[i] for i in range(num_words)]

        if method == 'tsne':
            reducer = TSNE(n_components=2, perplexity=30, random_state=42, init='pca')
        elif method == 'pca':
            reducer = PCA(n_components=2)
        else:
            return "Unknown decomposition method."

        projection = reducer.fit_transform(vectors_to_plot)

        plt.figure(figsize=(14, 10))
        plt.scatter(projection[:, 0], projection[:, 1], alpha=0.6, color='steelblue')

        for i, word in enumerate(words_to_plot):
            plt.annotate(word, xy=(projection[i, 0], projection[i, 1]),
                         xytext=(5, 2), textcoords='offset points',
                         ha='right', va='bottom', fontsize=8)

        plt.title(f"Word2Vec Embeddings ({method.upper()}) - {num_words} words")
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.show()


def _spearman_correlation(x, y):
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    def _rankdata(arr):
        order = np.argsort(arr)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(arr) + 1, dtype=np.float64)
        i = 0
        while i < len(arr):
            j = i + 1
            while j < len(arr) and arr[order[j]] == arr[order[i]]:
                j += 1
            if j > i + 1:
                avg_rank = np.mean(np.arange(i + 1, j + 1, dtype=np.float64))
                for k in range(i, j):
                    ranks[order[k]] = avg_rank
            i = j
        return ranks

    rx = _rankdata(x)
    ry = _rankdata(y)

    d = rx - ry
    n = len(x)
    rho = 1.0 - (6.0 * np.sum(d ** 2)) / (n * (n ** 2 - 1))
    return rho