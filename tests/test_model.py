import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import Model
from src.optim import SGD, AdaGrad, SGLD


def test_model_shapes():

    vocab_size = 100
    embedding_dim = 16
    batch_size = 32
    neg_samples = 5

    model = Model(vocab_size=vocab_size, embedding_dim=embedding_dim)

    centers = np.random.randint(0, vocab_size, size=batch_size)
    contexts = np.random.randint(0, vocab_size, size=batch_size)
    negatives = np.random.randint(0, vocab_size, size=(batch_size, neg_samples))

    #forward pass
    v_c, v_ctx, v_neg, pos_prob, neg_prob = model.forward_pass(centers, contexts, negatives)

    assert pos_prob.shape == (batch_size,), f"pos_prob shape: {pos_prob.shape}"
    assert neg_prob.shape == (batch_size, neg_samples), f"neg_prob shape: {neg_prob.shape}"

    #backward pass
    grad_in, grad_out_ctx, grad_out_neg = model.backward_pass(pos_prob, neg_prob, v_c, v_ctx, v_neg)

    assert grad_in.shape == (batch_size, embedding_dim)
    assert grad_out_ctx.shape == (batch_size, embedding_dim)
    assert grad_out_neg.shape == (batch_size, neg_samples, embedding_dim)

    print("[PASS] test_model_shapes")


def test_gradient_check():

    np.random.seed(5)
    vocab_size = 200
    embedding_dim = 8
    batch_size = 4
    neg_samples = 3

    model = Model(vocab_size=vocab_size, embedding_dim=embedding_dim)

    centers = np.random.randint(0, vocab_size, size=batch_size)
    contexts = np.random.randint(0, vocab_size, size=batch_size)
    negatives = np.random.randint(0, vocab_size, size=(batch_size, neg_samples))

    results = model.gradient_check(centers, contexts, negatives, num_checks=10)

    for param_name, rel_error in results.items():
        assert rel_error < 1e-4, \
            f"Gradient check FAILED for {param_name}: rel_error={rel_error:.2e}"
        print(f"  {param_name}: rel_error = {rel_error:.2e} [OK]")

    print("[PASS] test_gradient_check")


def test_sgd_reduces_loss():
    
    np.random.seed(6)
    vocab_size = 50
    model = Model(vocab_size=vocab_size, embedding_dim=16)
    optimizer = SGD(model, lr=0.1)

    centers = np.random.randint(0, vocab_size, size=16)
    contexts = np.random.randint(0, vocab_size, size=16)
    negatives = np.random.randint(0, vocab_size, size=(16, 5))

    initial_loss = model.train_step(centers, contexts, negatives, optimizer)
    for _ in range(50):
        loss = model.train_step(centers, contexts, negatives, optimizer)
    assert loss < initial_loss, f"SGD did not reduce loss: {initial_loss:.4f} -> {loss:.4f}"
    print(f"  SGD: {initial_loss:.4f} -> {loss:.4f}")
    print("[PASS] test_sgd_reduces_loss")


def test_adagrad_reduces_loss():
    
    np.random.seed(7)
    vocab_size = 50
    model = Model(vocab_size=vocab_size, embedding_dim=16)
    optimizer = AdaGrad(model, lr=0.5)

    centers = np.random.randint(0, vocab_size, size=16)
    contexts = np.random.randint(0, vocab_size, size=16)
    negatives = np.random.randint(0, vocab_size, size=(16, 5))

    initial_loss = model.train_step(centers, contexts, negatives, optimizer)
    for _ in range(50):
        loss = model.train_step(centers, contexts, negatives, optimizer)
    assert loss < initial_loss, f"AdaGrad did not reduce loss: {initial_loss:.4f} -> {loss:.4f}"
    print(f"  AdaGrad: {initial_loss:.4f} -> {loss:.4f}")
    print("[PASS] test_adagrad_reduces_loss")


def test_np_add_at_correctness():

    vocab_size = 10
    embedding_dim = 4
    model = Model(vocab_size=vocab_size, embedding_dim=embedding_dim)

    model.W_out = (np.random.rand(vocab_size, embedding_dim) - 0.5) / embedding_dim


    centers = np.array([0, 0, 0, 1], dtype=np.int32)
    contexts = np.array([2, 3, 4, 5], dtype=np.int32)
    negatives = np.random.randint(2, vocab_size, size=(4, 2))

    original_w_in_0 = model.W_in[0].copy()

    optimizer = SGD(model, lr=0.01)
    model.train_step(centers, contexts, negatives, optimizer)

    delta = model.W_in[0] - original_w_in_0
    assert np.any(np.abs(delta) > 1e-10), "No update applied"
    print("[PASS] test_np_add_at_correctness")

def test_sgld_reduces_loss():
    np.random.seed(10)
    vocab_size = 50
    model = Model(vocab_size=vocab_size, embedding_dim=16)
    #burn in epochs = 0
    optimizer = SGLD(model, lr=0.1, temperature=1e-5,
                     burn_in_epochs=0, total_steps=200)

    centers = np.random.randint(0, vocab_size, size=16)
    contexts = np.random.randint(0, vocab_size, size=16)
    negatives = np.random.randint(0, vocab_size, size=(16, 5))

    initial_loss = model.train_step(centers, contexts, negatives, optimizer)
    for _ in range(50):
        loss = model.train_step(centers, contexts, negatives, optimizer)
    assert loss < initial_loss, f"SGLD did not reduce loss: {initial_loss:.4f} -> {loss:.4f}"
    print(f"  SGLD: {initial_loss:.4f} -> {loss:.4f}")
    print("[PASS] test_sgld_reduces_loss")


def test_sgld_noise_active():

    np.random.seed(67)
    vocab_size = 50
    embedding_dim = 16

    model_sgd = Model(vocab_size=vocab_size, embedding_dim=embedding_dim)
    model_sgld = Model(vocab_size=vocab_size, embedding_dim=embedding_dim)
    model_sgld.W_in = model_sgd.W_in.copy()
    model_sgld.W_out = model_sgd.W_out.copy()

    opt_sgd = SGD(model_sgd, lr=0.1)
    opt_sgld = SGLD(model_sgld, lr=0.1, temperature=1e-2,
                    burn_in_epochs=0, total_steps=200)

    centers = np.random.randint(0, vocab_size, size=16)
    contexts = np.random.randint(0, vocab_size, size=16)
    negatives = np.random.randint(0, vocab_size, size=(16, 5))

    for _ in range(20):
        model_sgd.train_step(centers, contexts, negatives, opt_sgd)
        model_sgld.train_step(centers, contexts, negatives, opt_sgld)

    diff = np.linalg.norm(model_sgd.W_in - model_sgld.W_in)
    assert diff > 1e-6, f"SGLD embeddings identical to SGD (diff={diff:.2e})"
    print(f"  W_in divergence (SGLD vs SGD): {diff:.4f}")
    print("[PASS] test_sgld_noise_active")


def test_sgld_snapshots():
    np.random.seed(12)
    vocab_size = 30
    model = Model(vocab_size=vocab_size, embedding_dim=8)

    total_steps = 50

    optimizer = SGLD(model, lr=0.05, total_steps=total_steps,
                     burn_in_epochs=1, snapshot_interval=5, max_snapshots=100,
                     temperature=1e-4, centroid_every=1)

    centers = np.random.randint(0, vocab_size, size=8)
    contexts = np.random.randint(0, vocab_size, size=8)
    negatives = np.random.randint(0, vocab_size, size=(8, 3))

    for _ in range(total_steps):
        model.train_step(centers, contexts, negatives, optimizer, current_epoch=1)

    snapshots = optimizer.get_snapshots()
    # Snapshots at steps 5, 10, ... 50 = 10 snapshots
    assert len(snapshots) >= 5, f"Too few snapshots: {len(snapshots)}"
    assert snapshots[0].shape == (vocab_size, 8), \
        f"Snapshot shape wrong: {snapshots[0].shape}"
    # Snapshots should differ (noise is active)
    assert not np.allclose(snapshots[0], snapshots[-1]), \
        "First and last snapshot are identical — noise not working"
    print(f"  Collected {len(snapshots)} snapshots")
    print("[PASS] test_sgld_snapshots")


def test_sgld_centroid():

    np.random.seed(13)
    vocab_size = 30
    model = Model(vocab_size=vocab_size, embedding_dim=8)

    optimizer = SGLD(model, lr=0.05, total_steps=100,
                     burn_in_epochs=0, centroid_every=1,
                     temperature=1e-3, snapshot_interval=0)

    centers = np.random.randint(0, vocab_size, size=8)
    contexts = np.random.randint(0, vocab_size, size=8)
    negatives = np.random.randint(0, vocab_size, size=(8, 3))

    for _ in range(30):
        model.train_step(centers, contexts, negatives, optimizer)

    centroid = optimizer.get_centroid()
    assert centroid is not None, "Centroid not populated"
    assert centroid.shape == model.W_in.shape, \
        f"Centroid shape wrong: {centroid.shape}"

    diff = np.linalg.norm(centroid - model.W_in)
    assert diff > 1e-6, f"Centroid equals raw W_in (diff={diff:.2e})"
    print(f"  Centroid samples: {optimizer.centroid_count}, divergence from W_in: {diff:.4f}")
    print("[PASS] test_sgld_centroid")


if __name__ == "__main__":
    print("=" * 50)
    print("Running Word2Vec tests")
    print("=" * 50)

    test_model_shapes()
    test_gradient_check()
    test_sgd_reduces_loss()
    test_adagrad_reduces_loss()
    test_np_add_at_correctness()
    test_sgld_reduces_loss()
    test_sgld_noise_active()
    test_sgld_snapshots()
    test_sgld_centroid()

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)
