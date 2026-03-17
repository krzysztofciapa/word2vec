import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

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
        
        nearest_ids = np.argsort(cosine_similarities)[::-1][1:k+1]
        
        return [(self.id2word[idx], cosine_similarities[idx]) for idx in nearest_ids]
    


    def plot_embeddings(self, num_words=300, method='tsne'):

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
                         
        plt.title(f"Embeddins in Word2Vec ({method.upper()}) - {num_words} words")
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.show()