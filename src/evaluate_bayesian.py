import numpy as np


class BayesianEvaluator:

    def __init__(self, filepath):

        data = np.load(filepath, allow_pickle=True)
        self.word2id = data['word2id'].item()
        self.id2word = {idx: word for word, idx in self.word2id.items()}
        self.point_embeddings = data['embeddings']

        if 'snapshots' in data:
            self.snapshots = data['snapshots']  # shape: T x vocab_size x dim
            print(f"Loaded {self.snapshots.shape[0]} posterior snapshots "
                  f"(vocab={self.snapshots.shape[1]}, dim={self.snapshots.shape[2]})")
        else:
            self.snapshots = None
            print("WARNING: No SGLD snapshots found. Train with --optimizer sgld first.")

        # normalized snapshots
        self._norm_snapshots = None
        if self.snapshots is not None:
            norms = np.linalg.norm(self.snapshots, axis=2, keepdims=True)
            norms[norms == 0] = 1e-10
            self._norm_snapshots = self.snapshots / norms

    def similarity_with_uncertainty(self, word1, word2):

        if self._norm_snapshots is None:
            print("No snapshots available.")
            return None

        if word1 not in self.word2id or word2 not in self.word2id:
            missing = [w for w in [word1, word2] if w not in self.word2id]
            print(f"Word(s) not in vocabulary: {missing}")
            return None

        id1 = self.word2id[word1]
        id2 = self.word2id[word2]

        # cosine similarity across all T snapshots
        vecs1 = self._norm_snapshots[:, id1, :]
        vecs2 = self._norm_snapshots[:, id2, :]
        sims = np.sum(vecs1 * vecs2, axis=1) 

        return float(np.mean(sims)), float(np.std(sims)), sims

    def evaluate_pairs(self, word_pairs):

        if self._norm_snapshots is None:
            print("No snapshots available for Bayesian evaluation.")
            return []

        results = []
        print(f"\n{'='*65}")
        print(f"BAYESIAN SIMILARITY (SGLD, {self._norm_snapshots.shape[0]} posterior samples)")
        print(f"{'='*65}")

        for w1, w2 in word_pairs:
            result = self.similarity_with_uncertainty(w1, w2)
            if result is not None:
                mean_sim, std_sim, _ = result
                results.append((w1, w2, mean_sim, std_sim))
                print(f"  {w1:>12s} ↔ {w2:<12s}  "
                      f"sim = {mean_sim:+.4f} ± {std_sim:.4f}")
            else:
                print(f"  {w1:>12s} ↔ {w2:<12s}  [skipped - OOV]")

        print(f"{'='*65}\n")
        return results

    def uncertainty_vs_frequency(self, plot_path=None, num_words=500):

        if self._norm_snapshots is None:
            print("No snapshots available.")
            return None, None

        all_ids = list(self.id2word.keys())
        if len(all_ids) > num_words:
            sampled_ids = np.random.choice(all_ids, size=num_words, replace=False)
        else:
            sampled_ids = all_ids

        log_freq_ranks = []
        uncertainties = []

        for wid in sampled_ids:
           
            vecs = self._norm_snapshots[:, wid, :] 
            mean_vec = np.mean(vecs, axis=0)
            mean_norm = np.linalg.norm(mean_vec)
            if mean_norm < 1e-10:
                continue
            mean_vec_normed = mean_vec / mean_norm
            cos_to_mean = np.dot(vecs, mean_vec_normed)  # (T,)
            uncertainty = float(np.std(cos_to_mean))

            log_freq_ranks.append(np.log10(wid + 1))
            uncertainties.append(uncertainty)

        log_freq_ranks = np.array(log_freq_ranks)
        uncertainties = np.array(uncertainties)

        if len(log_freq_ranks) > 1:
            correlation = np.corrcoef(log_freq_ranks, uncertainties)[0, 1]
        else:
            correlation = 0.0

        print(f"\n{'='*60}")
        print(f"UNCERTAINTY vs FREQUENCY RANK ANALYSIS")
        print(f"{'='*60}")
        print(f"  Words analyzed:    {len(log_freq_ranks)}")
        print(f"  Pearson r:         {correlation:.4f}")
        print(f"  Expected:          positive (higher rank/rarer → more uncertain)")
        print(f"{'='*60}\n")
        
        if plot_path:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 6))
            plt.scatter(log_freq_ranks, uncertainties, alpha=0.5, color='#e74c3c')
            
            # line of best fit
            z = np.polyfit(log_freq_ranks, uncertainties, 1)
            p = np.poly1d(z)
            plt.plot(log_freq_ranks, p(log_freq_ranks), "k--", alpha=0.8)
            
            plt.title('Bayesian Uncertainty vs. Word Frequency Rank', fontsize=14)
            plt.xlabel('Log10(Vocabulary Rank) (Higher = Rarer Word)', fontsize=12)
            plt.ylabel('Embedding Uncertainty (Standard Deviation)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300)
            plt.close()

        return log_freq_ranks, uncertainties, correlation
