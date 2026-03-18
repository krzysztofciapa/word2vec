import numpy as np

def sigmoid(x):
    x = np.clip(x, -15.0, 15.0)
    return 1.0 / (1.0 + np.exp(-x))

class Model:
    def __init__(self, vocab_size, embedding_dim=100):

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        self.W_in = (np.random.rand(vocab_size, embedding_dim) - 0.5) / embedding_dim
        self.W_out = np.zeros((vocab_size, embedding_dim))
        self.sig_max = 6.0
        self.sig_table_size = 1000
        self.sig_table = self._build_sigmoid_table()

    def _build_sigmoid_table(self):
        x = np.linspace(-self.sig_max, self.sig_max, self.sig_table_size)
        return 1.0 / (1.0 + np.exp(-x))
    
    def fast_sigmoid(self, x):
        idx = (x + self.sig_max) * (self.sig_table_size / (2.0 * self.sig_max))
        idx = np.clip(idx, 0, self.sig_table_size - 1).astype(int)
        return self.sig_table[idx]

    def forward_pass(self, centers, contexts, negatives):

        v_c = self.W_in[centers]       # dimension BxD
        v_ctx = self.W_out[contexts]   # dimenstion BxK
        v_neg = self.W_out[negatives]  # dimension BxKxD

        #positive signal
        pos_dot = np.sum(v_c * v_ctx, axis=1)
        pos_prob = self.fast_sigmoid(pos_dot) 

        #negative signal
        neg_dot = np.einsum('bd,bkd->bk', v_c, v_neg, optimize=True)
        neg_prob = self.fast_sigmoid(neg_dot)

        return v_c, v_ctx, v_neg, pos_prob, neg_prob
    

    def compute_loss(self, pos_prob, neg_prob):

        pos_loss = -np.log(pos_prob + 1e-10)
        neg_loss = -np.sum(np.log(1.0 - neg_prob + 1e-10), axis=1)
        
        batch_loss = np.mean(pos_loss + neg_loss)
        return batch_loss

    def backward_pass(self, pos_prob, neg_prob, v_c, v_ctx, v_neg):

        pos_error = pos_prob - 1.0 
        neg_error = neg_prob - 0.0


        grad_W_out_ctx = pos_error[:, np.newaxis] * v_c
        grad_W_out_neg = neg_error[:, :, np.newaxis] * v_c[:, np.newaxis, :]

        grad_W_in = (pos_error[:, np.newaxis] * v_ctx) + \
                    np.einsum('bk,bkd->bd', neg_error, v_neg, optimize=True)

        return grad_W_in, grad_W_out_ctx, grad_W_out_neg

    def train_step(self, centers, contexts, negatives, optimizer):

        v_c, v_ctx, v_neg, pos_prob, neg_prob = self.forward_pass(centers, contexts, negatives)
        
        loss = self.compute_loss(pos_prob, neg_prob)
        
        grad_in, grad_out_ctx, grad_out_neg = self.backward_pass(pos_prob, neg_prob, v_c, v_ctx, v_neg)
        
        optimizer.step(centers, contexts, negatives, grad_in, grad_out_ctx, grad_out_neg)
        
        return loss
    

    def save_weights(self, filepath, word2id):

        np.savez_compressed( filepath, embeddings=self.W_in, word2id=word2id )