import numpy as np


class SGD:

    def __init__(self, model, lr=0.025, min_lr=0.0001, total_steps=None):
        self.model = model
        self.initial_lr = lr
        self.current_lr = lr
        self.min_lr = min_lr
        self.total_steps = total_steps
        self.current_step = 0

    def step(self, center_ids, context_ids, negative_ids, grad_W_in, grad_W_out_ctx, grad_W_out_neg):
        if self.total_steps is not None and self.total_steps > 0:
            progress = self.current_step / self.total_steps
            self.current_lr = max(self.min_lr, self.initial_lr * (1.0 - progress))
            self.current_step += 1

        lr = self.current_lr

        # np.add.at handles duplicate indices
        np.add.at(self.model.W_in, center_ids, -lr * grad_W_in)
        np.add.at(self.model.W_out, context_ids, -lr * grad_W_out_ctx)

        flat_neg_ids = negative_ids.flatten()
        flat_grad_neg = grad_W_out_neg.reshape(-1, self.model.embedding_dim)
        np.add.at(self.model.W_out, flat_neg_ids, -lr * flat_grad_neg)


class AdaGrad:


    def __init__(self, model, lr=0.025, eps=1e-8, decay=0.99):
        self.model = model
        self.current_lr = lr
        self.eps = eps
        self.decay = decay

        # per-parameter squared gradient accumulators
        self.G_in = np.zeros_like(model.W_in)
        self.G_out = np.zeros_like(model.W_out)

    def step(self, center_ids, context_ids, negative_ids, grad_W_in, grad_W_out_ctx, grad_W_out_neg):
        # W_in center embeddings
        np.add.at(self.G_in, center_ids, grad_W_in ** 2)
        adaptive_lr_in = self.current_lr / np.sqrt(self.G_in[center_ids] + self.eps)
        np.add.at(self.model.W_in, center_ids, -adaptive_lr_in * grad_W_in)

        # W_out context embeddings
        np.add.at(self.G_out, context_ids, grad_W_out_ctx ** 2)
        adaptive_lr_ctx = self.current_lr / np.sqrt(self.G_out[context_ids] + self.eps)
        np.add.at(self.model.W_out, context_ids, -adaptive_lr_ctx * grad_W_out_ctx)

        # W_out negative embeddings
        flat_neg_ids = negative_ids.flatten()
        flat_grad_neg = grad_W_out_neg.reshape(-1, self.model.embedding_dim)
        np.add.at(self.G_out, flat_neg_ids, flat_grad_neg ** 2)
        adaptive_lr_neg = self.current_lr / np.sqrt(self.G_out[flat_neg_ids] + self.eps)
        np.add.at(self.model.W_out, flat_neg_ids, -adaptive_lr_neg * flat_grad_neg)



class SGLD:


    def __init__(self, model, lr=0.025, min_lr=0.0001, total_steps=None,
                 temperature=1e-4, burn_in_epochs=1,
                 snapshot_interval=5000, max_snapshots=20,
                 centroid_every=100):
        self.model = model
        self.initial_lr = lr
        self.current_lr = lr
        self.min_lr = min_lr
        self.total_steps = total_steps
        self.current_step = 0
        self.rng = np.random.default_rng(42)

        #Langevin parameters
        self.temperature = temperature
        self.burn_in_epochs = burn_in_epochs

        # posterior collection
        self.snapshot_interval = snapshot_interval
        self.max_snapshots = max_snapshots
        self.snapshots = []

        self.centroid_every = centroid_every
        self.centroid = None     # running mean of W_in (Welford method)
        self.centroid_count = 0 

 
    def step(self, center_ids, context_ids, negative_ids,
             grad_W_in, grad_W_out_ctx, grad_W_out_neg, current_epoch):

        # learning-rate schedule (linear decay, same as SGD)
        if self.total_steps is not None and self.total_steps > 0:
            progress = self.current_step / self.total_steps
            self.current_lr = max(self.min_lr, self.initial_lr * (1.0 - progress))

        lr = self.current_lr
        self.current_step += 1

        #gradient step
        np.add.at(self.model.W_in, center_ids, -lr * grad_W_in)
        np.add.at(self.model.W_out, context_ids, -lr * grad_W_out_ctx)

        flat_neg_ids = negative_ids.flatten()
        flat_grad_neg = grad_W_out_neg.reshape(-1, self.model.embedding_dim)
        np.add.at(self.model.W_out, flat_neg_ids, -lr * flat_grad_neg)

        # Langevin noise injection (after burn-in only)
        if current_epoch >= self.burn_in_epochs:
            noise_std = np.sqrt(2.0 * lr * self.temperature)

            # W_in — unique centers
            unique_in = np.unique(center_ids)
            self.model.W_in[unique_in] += noise_std * self.rng.standard_normal(
                (len(unique_in), self.model.embedding_dim))

            # W_out — unique context + negative ids
            unique_out = np.unique(np.concatenate([context_ids, flat_neg_ids]))
            self.model.W_out[unique_out] += noise_std * self.rng.standard_normal(
                (len(unique_out), self.model.embedding_dim))

            #centroid update
            if self.centroid_every > 0 and self.current_step % self.centroid_every == 0:
                self.centroid_count += 1
                if self.centroid is None:
                    self.centroid = self.model.W_in.copy()
                else:
                    self.centroid += (self.model.W_in - self.centroid) / self.centroid_count

            #snapshot collection
            if (self.snapshot_interval > 0
                    and self.current_step % self.snapshot_interval == 0
                    and len(self.snapshots) < self.max_snapshots):
                self.snapshots.append(self.model.W_in.copy())

    def get_centroid(self):
        return self.centroid

    def get_snapshots(self):
        return self.snapshots