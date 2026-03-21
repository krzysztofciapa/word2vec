import numpy as np


def sigmoid(x):
    x = np.clip(x, -15.0, 15.0)
    return 1.0 / (1.0 + np.exp(-x))


class Model:
    def __init__(self, vocab_size, embedding_dim=100):

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Xavier initialization
        self.W_in = np.random.randn(vocab_size, embedding_dim) * (1.0 / np.sqrt(embedding_dim))
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

    def forward_pass(self, centers, contexts, negatives, exact_sigmoid=False):

        v_c = self.W_in[centers]       # B x D
        v_ctx = self.W_out[contexts]   # B x D
        v_neg = self.W_out[negatives]  # B x K x D

        sig_fn = sigmoid if exact_sigmoid else self.fast_sigmoid

        # positive signal
        pos_dot = np.sum(v_c * v_ctx, axis=1)
        pos_prob = sig_fn(pos_dot)

        # negative signal
        neg_dot = np.einsum('bd,bkd->bk', v_c, v_neg, optimize=True)
        neg_prob = sig_fn(neg_dot)

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

    def train_step(self, centers, contexts, negatives, optimizer, current_epoch=0):

        v_c, v_ctx, v_neg, pos_prob, neg_prob = self.forward_pass(centers, contexts, negatives)

        loss = self.compute_loss(pos_prob, neg_prob)

        grad_in, grad_out_ctx, grad_out_neg = self.backward_pass(pos_prob, neg_prob, v_c, v_ctx, v_neg)

        # optimizers that need to know the current epoch
        import inspect
        sig = inspect.signature(optimizer.step)
        if 'current_epoch' in sig.parameters:
            optimizer.step(centers, contexts, negatives, grad_in, grad_out_ctx, grad_out_neg, current_epoch=current_epoch)
        else:
            optimizer.step(centers, contexts, negatives, grad_in, grad_out_ctx, grad_out_neg)

        return loss

    def gradient_check(self, centers, contexts, negatives, eps=1e-5, num_checks=5):
  
        # exact sigmoid
        v_c, v_ctx, v_neg, pos_prob, neg_prob = self.forward_pass(
            centers, contexts, negatives, exact_sigmoid=True
        )
        grad_in, grad_out_ctx, grad_out_neg = self.backward_pass(
            pos_prob, neg_prob, v_c, v_ctx, v_neg
        )

        B = len(centers)
        results = {}

        def _numerical_grad(matrix_name, row_idx, col_idx):
            matrix = self.W_in if matrix_name == 'W_in' else self.W_out
            original = matrix[row_idx, col_idx].copy()

            matrix[row_idx, col_idx] = original + eps
            _, _, _, pp, np_ = self.forward_pass(centers, contexts, negatives, exact_sigmoid=True)
            loss_plus = self.compute_loss(pp, np_)

            matrix[row_idx, col_idx] = original - eps
            _, _, _, pp, np_ = self.forward_pass(centers, contexts, negatives, exact_sigmoid=True)
            loss_minus = self.compute_loss(pp, np_)

            matrix[row_idx, col_idx] = original
            return (loss_plus - loss_minus) / (2.0 * eps)

        # check W_in gradients
        errors_in = []
        checked_in = set()
        for _ in range(num_checks * 3):
            if len(errors_in) >= num_checks:
                break
            b = np.random.randint(B)
            d = np.random.randint(self.embedding_dim)
            idx = centers[b]
            key = (idx, d)
            if key in checked_in:
                continue
            checked_in.add(key)

            #sum grad_in over all positions where centers == idx
            analytic = np.sum(grad_in[centers == idx, d]) / B
            numerical = _numerical_grad('W_in', idx, d)

            rel_error = abs(numerical - analytic) / (abs(numerical) + abs(analytic) + 1e-15)
            errors_in.append(rel_error)

        results['W_in'] = np.mean(errors_in) if errors_in else 0.0

        # check W_out gradients (context + negative combined)
        errors_out = []
        checked_out = set()
        for _ in range(num_checks * 3):
            if len(errors_out) >= num_checks:
                break
            b = np.random.randint(B)
            d = np.random.randint(self.embedding_dim)
            idx = contexts[b]
            key = (idx, d)
            if key in checked_out:
                continue
            checked_out.add(key)


            analytic = 0.0
            analytic += np.sum(grad_out_ctx[contexts == idx, d])
            neg_mask = (negatives == idx)
            if np.any(neg_mask):
                analytic += np.sum(grad_out_neg[neg_mask, d])
            analytic /= B

            numerical = _numerical_grad('W_out', idx, d)

            rel_error = abs(numerical - analytic) / (abs(numerical) + abs(analytic) + 1e-15)
            errors_out.append(rel_error)

        results['W_out'] = np.mean(errors_out) if errors_out else 0.0

        return results

    def save_weights(self, filepath, word2id, snapshots=None, W_in_override=None):
        embeddings = W_in_override if W_in_override is not None else self.W_in
        save_dict = dict(embeddings=embeddings, word2id=word2id)
        if snapshots is not None and len(snapshots) > 0:
            save_dict['snapshots'] = np.stack(snapshots)
        np.savez_compressed(filepath, **save_dict)