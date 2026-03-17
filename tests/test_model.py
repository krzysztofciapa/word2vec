import numpy as np
from src.model import Model

def test_model_shapes():

    vocab_size = 100
    embedding_dim = 16
    batch_size = 32
    neg_samples = 5
    
    
    model = Model(vocab_size=vocab_size, embedding_dim=embedding_dim)
    
    centers = np.random.randint(0, vocab_size, size=batch_size)
    contexts = np.random.randint(0, vocab_size, size=batch_size)
    negatives = np.random.randint(0, vocab_size, size=(batch_size, neg_samples))
    
    #Forward pass validation
    v_c, v_ctx, v_neg, pos_prob, neg_prob = model.forward_pass(centers, contexts, negatives)
    
    assert pos_prob.shape == (batch_size,), "Dimension error: Positive probability array"
    assert neg_prob.shape == (batch_size, neg_samples), "Dimension error: Negative probability array"
    
    #Backpropagation validation
    grad_in, grad_out_ctx, grad_out_neg = model.backward_pass(pos_prob, neg_prob, v_c, v_ctx, v_neg)
    
    assert grad_in.shape == (batch_size, embedding_dim), "Error: Invalid input weight gradient (W_in)"
    assert grad_out_ctx.shape == (batch_size, embedding_dim), "Error: Invalid context weight gradient (W_out)"
    assert grad_out_neg.shape == (batch_size, neg_samples, embedding_dim), "Error: Invalid negative noise tensor (W_out)"