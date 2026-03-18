class SGD:
    def __init__(self, model, lr=0.025, min_lr=0.0001, total_steps=None):
        """
        Stochastic Gradient Descent optimizer with linear learning rate decay.
        :param model: The Word2Vec model instance.
        :param lr: Initial learning rate.
        :param min_lr: Minimum learning rate after decay.
        :param total_steps: Total number of training steps (batches) for LR decay.
        """
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

        self.model.W_in[center_ids] -= self.current_lr * grad_W_in
        self.model.W_out[context_ids] -= self.current_lr * grad_W_out_ctx
        
        flat_neg_ids = negative_ids.flatten()
        flat_grad_neg = grad_W_out_neg.reshape(-1, self.model.embedding_dim)
        
        self.model.W_out[flat_neg_ids] -= self.current_lr * flat_grad_neg
