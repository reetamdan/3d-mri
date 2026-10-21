"""Seeded cluster-bootstrap intervals and explicitly scoped feature diagnostics."""
import numpy as np


def bootstrap_interval(values, groups=None, seed=42, resamples=2000, confidence=.95):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Bootstrap requires a finite nonempty vector')
    if resamples < 100 or not 0 < confidence < 1:
        raise ValueError('Use >=100 resamples and confidence in (0,1)')
    g = np.arange(len(x)).astype(str) if groups is None else np.asarray(groups).astype(str)
    if g.shape != x.shape:
        raise ValueError('One cluster identity is required per observation')
    unique = np.unique(g)
    clusters = [x[g == key] for key in unique]
    rng = np.random.default_rng(seed)
    estimates = np.array([np.concatenate([clusters[i] for i in rng.integers(len(clusters), size=len(clusters))]).mean()
                          for _ in range(resamples)])
    alpha = (1-confidence)/2
    return dict(mean=float(x.mean()), lower=float(np.quantile(estimates, alpha)),
                upper=float(np.quantile(estimates, 1-alpha)), confidence=confidence,
                observations=len(x), clusters=len(unique), resamples=resamples, seed=seed,
                method='cluster bootstrap' if groups is not None else 'observation bootstrap',
                uncertainty_available=len(unique) >= 2)


def feature_distribution_metrics(real, generated):
    """Fréchet feature distance and unbiased polynomial-kernel MMD (KID).

    The caller must specify/validate the extractor. These are not automatically
    clinical or medical-FID scores. No pretrained weights are downloaded.
    """
    a, b = np.asarray(real, dtype=np.float64), np.asarray(generated, dtype=np.float64)
    if (a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or not a.shape[1]
            or min(len(a), len(b)) < 2 or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError('Feature metrics require finite N by D arrays, >=2 observations, and matching dimensions')
    ca, cb = np.atleast_2d(np.cov(a, rowvar=False)), np.atleast_2d(np.cov(b, rowvar=False))
    eigenvalues, vectors = np.linalg.eigh(ca)
    half = (vectors * np.sqrt(np.maximum(eigenvalues, 0))) @ vectors.T
    product = half @ cb @ half
    covariance_trace = np.sqrt(np.maximum(np.linalg.eigvalsh((product+product.T)/2), 0)).sum()
    distance = np.square(a.mean(0)-b.mean(0)).sum()+np.trace(ca)+np.trace(cb)-2*covariance_trace
    kaa, kbb, kab = (a@a.T/a.shape[1]+1)**3, (b@b.T/b.shape[1]+1)**3, (a@b.T/a.shape[1]+1)**3
    kid = ((kaa.sum()-np.trace(kaa))/(len(a)*(len(a)-1))
           +(kbb.sum()-np.trace(kbb))/(len(b)*(len(b)-1))-2*kab.mean())
    return dict(frechet_feature_distance=float(max(distance, 0)), polynomial_kid_unbiased=float(kid),
                real_count=len(a), generated_count=len(b), feature_dimension=a.shape[1],
                covariance_rank_limited=min(len(a), len(b)) <= a.shape[1],
                interpretation='Extractor-dependent feature diagnostics. KID can be negative; small cohorts have high uncertainty.')
