import numpy as np

from sciintel.clustering import _calculate_metrics, gpu_cosine_distances


def test_cosine_distance_matrix_is_symmetric_and_normalized():
    vectors = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    distances, backend = gpu_cosine_distances(vectors, device="cpu")
    assert backend == "cpu"
    assert distances.shape == (3, 3)
    assert np.allclose(distances, distances.T, atol=1e-6)
    assert np.allclose(np.diag(distances), 0.0, atol=1e-7)
    assert distances[0, 1] < distances[0, 2]


def test_metrics_report_noise_separately_from_clusters():
    distances = np.asarray(
        [
            [0.0, 0.05, 0.9, 0.95],
            [0.05, 0.0, 0.92, 0.94],
            [0.9, 0.92, 0.0, 0.06],
            [0.95, 0.94, 0.06, 0.0],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, -1], dtype=np.int64)
    metrics = _calculate_metrics(distances, labels, ["a", "a", "b", "b"])
    assert metrics["clusters"] == 2
    assert metrics["noise_count"] == 1
    assert metrics["noise_fraction"] == 0.25
    assert metrics["ari_domains"] is not None
    assert metrics["nmi_domains"] is not None


def test_umap_projection_uses_reproducible_parameters(monkeypatch):
    import sys
    from types import SimpleNamespace

    from sciintel.clustering import umap_projection

    captured = {}

    class FakeUMAP:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def fit_transform(self, vectors):
            return np.zeros((len(vectors), captured["n_components"]), dtype=np.float64)

    monkeypatch.setitem(sys.modules, "umap", SimpleNamespace(UMAP=FakeUMAP))
    vectors = np.ones((8, 16), dtype=np.float32)
    result = umap_projection(vectors, n_components=10, n_neighbors=30, min_dist=0.0)

    assert result.shape == (8, 10)
    assert result.dtype == np.float32
    assert captured["n_neighbors"] == 7
    assert captured["n_components"] == 10
    assert captured["metric"] == "cosine"
    assert captured["random_state"] == 42
    assert captured["n_jobs"] == 1
