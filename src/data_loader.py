import re
from collections import Counter, deque
import numpy as np

class TextStreamer:
    def __init__(self, file_path, limit_lines=None):
        self.file_path = file_path
        self.limit_lines = limit_lines

    def __iter__(self):
        with open(self.file_path, 'r', encoding='utf-8') as file:
            for line_number, line in enumerate(file):
                if self.limit_lines is not None and line_number >= self.limit_lines:
                    break
                    
                tokens = re.findall(r'[a-z]+', line.lower())
                for token in tokens:
                    yield token
class Vocabulary:
    def __init__(self, min_count=5, subsampling_t=1e-5):
        self.min_count = min_count
        self.subsampling_t = subsampling_t
        
        self.word2id = {}
        self.id2word = {}
        self.word_counts = {}
        self.total_words = 0
        self.discard_probs = {}

    def build_vocabulary(self, stream):

        raw_counts = Counter(stream)
        
        current_id = 0
        for word, count in raw_counts.items():
            if count >= self.min_count:
                self.word2id[word] = current_id
                self.id2word[current_id] = word
                self.word_counts[current_id] = count
                self.total_words += count
                current_id += 1
                
        self._calculate_discard_probabilities()

    def _calculate_discard_probabilities(self):
        for word_id, count in self.word_counts.items():
            frequency = count / self.total_words
            p_discard = max(0.0, 1.0 - np.sqrt(self.subsampling_t / frequency))
            self.discard_probs[word_id] = p_discard

    def __len__(self):
        return len(self.word2id)
    




class DataLoader:
    def __init__(self, text_stream, vocabulary, window_size=2, batch_size=128, neg_samples=5):

        self.stream = text_stream
        self.vocab = vocabulary
        self.window_size = window_size
        self.batch_size = batch_size
        self.neg_samples = neg_samples
        

        self.table_size = int(1e6)
        self.unigram_table = self._build_unigram_table()


    def _build_unigram_table(self):
        table = np.zeros(self.table_size, dtype=np.int32)
        
        word_ids = np.array(list(self.vocab.word_counts.keys()))
        pow_freq = np.array(list(self.vocab.word_counts.values())) ** 0.75
        
        total_pow = np.sum(pow_freq)
        probs = pow_freq / total_pow
        
        idx = 0
        cumulative_prob = probs[0]
        
        for i in range(self.table_size):
            table[i] = word_ids[idx] 
            if i / self.table_size > cumulative_prob:
                idx += 1
                if idx >= len(probs):
                    idx = len(probs) - 1
                cumulative_prob += probs[idx]
                
        return table

    def __iter__(self):
        span = 2 * self.window_size + 1
        buffer = deque(maxlen=span)
        
        centers = []
        contexts = []
        negatives = []
        
        for token in self.stream:
            if token not in self.vocab.word2id:
                continue
            word_id = self.vocab.word2id[token]
            
            p_discard = self.vocab.discard_probs.get(word_id, 0.0)
            if np.random.rand() < p_discard:
                continue
                
            buffer.append(word_id)
            
            if len(buffer) < span:
                continue
                
            center_word = buffer[self.window_size]
            
            dynamic_w = np.random.randint(1, self.window_size + 1)
            
            start_idx = self.window_size - dynamic_w
            end_idx = self.window_size + dynamic_w + 1
            
            context_words = [buffer[i] for i in range(start_idx, end_idx) if i != self.window_size]
            
            for context_word in context_words:
                centers.append(center_word)
                contexts.append(context_word)
                
                neg_indices = np.random.randint(0, self.table_size, size=self.neg_samples)
                negatives.append(self.unigram_table[neg_indices])
                
                if len(centers) == self.batch_size:
                    yield (
                        np.array(centers, dtype=np.int32), 
                        np.array(contexts, dtype=np.int32), 
                        np.array(negatives, dtype=np.int32)
                    )
                    centers, contexts, negatives = [], [], []

        if len(centers) > 0:
            yield (
                np.array(centers, dtype=np.int32), 
                np.array(contexts, dtype=np.int32), 
                np.array(negatives, dtype=np.int32)
            )

        span = 2 * self.window_size + 1
        buffer = deque(maxlen=span)
        
        centers = []
        contexts = []
        negatives = []
        
        for token in self.stream:

            if token not in self.vocab.word2id:
                continue
            word_id = self.vocab.word2id[token]
            
            p_discard = self.vocab.discard_probs.get(word_id, 0.0)
            if np.random.rand() < p_discard:
                continue
                
            buffer.append(word_id)
            
            if len(buffer) < span:
                continue
                
            center_word = buffer[self.window_size]
            context_words = [buffer[i] for i in range(span) if i != self.window_size]
            

            for context_word in context_words:
                centers.append(center_word)
                contexts.append(context_word)
                
                neg_indices = np.random.randint(0, self.table_size, size=self.neg_samples)
                negatives.append(self.unigram_table[neg_indices])
                
                if len(centers) == self.batch_size:
                    yield (
                        np.array(centers, dtype=np.int32), 
                        np.array(contexts, dtype=np.int32), 
                        np.array(negatives, dtype=np.int32)
                    )
                    centers, contexts, negatives = [], [], []

        if len(centers) > 0:
            yield (
                np.array(centers, dtype=np.int32), 
                np.array(contexts, dtype=np.int32), 
                np.array(negatives, dtype=np.int32)
            )