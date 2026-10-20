import numpy as np

SPLITS = ('train', 'validation', 'test')


def split_patients(patient_ids: list[str], ratios: dict, seed: int = 42) -> dict[str, str]:
    values = np.array([ratios[k] for k in SPLITS], dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0) or not np.isclose(values.sum(), 1):
        raise ValueError('Split ratios must be nonnegative and sum to 1')
    patients = sorted(set(patient_ids))
    np.random.default_rng(seed).shuffle(patients)
    exact = values * len(patients)
    counts = np.floor(exact).astype(int)
    for i in np.argsort(-(exact-counts), kind='stable')[:len(patients)-counts.sum()]:
        counts[i] += 1
    assignment = {}
    start = 0
    for split, count in zip(SPLITS, counts):
        assignment.update({pid: split for pid in patients[start:start+count]})
        start += count
    return assignment
