import warnings

import numpy as np
from scipy import optimize

from panicle.association.lrt import (
    fit_marker_lrt,
    fit_marker_lrt_prebuilt,
    fit_markers_lrt_batch_prebuilt,
)
from panicle.association.mlm import _calculate_neg_ml_likelihood


def test_fit_marker_lrt_returns_finite_effects() -> None:
    n = 6
    g = np.array([0, 1, 2, 0, 1, 2], dtype=np.float64)
    y = 2.0 + 0.8 * g + np.array([0.1, -0.2, 0.05, -0.1, 0.2, -0.05], dtype=np.float64)
    X = np.ones((n, 1), dtype=np.float64)
    eigenvals = np.ones(n, dtype=np.float64)

    def neg_ll(h2: float) -> float:
        return _calculate_neg_ml_likelihood(h2, y, X, eigenvals)

    result = optimize.minimize_scalar(
        neg_ll,
        bounds=(0.001, 0.999),
        method="bounded",
        options={"xatol": 1.22e-4, "maxiter": 100},
    )
    null_neg_loglik = result.fun if result.success else neg_ll(0.5)

    lrt_stat, p_value, beta, se = fit_marker_lrt(y, X, g, eigenvals, null_neg_loglik)

    assert 0.0 <= p_value <= 1.0
    assert np.isfinite(lrt_stat)
    assert np.isfinite(beta)
    assert np.isfinite(se)
    assert se > 0.0


def test_neg_ml_likelihood_stable_for_near_singular_design() -> None:
    n = 10
    x = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    X = np.column_stack([np.ones(n), x, x + 1e-10])
    y = 0.5 + 1.2 * x + np.array([0.01, -0.02, 0.01, -0.01, 0.02, -0.01, 0.01, -0.02, 0.01, -0.01])
    eigenvals = np.geomspace(1e-6, 2.0, n)

    val = _calculate_neg_ml_likelihood(0.5, y, X, eigenvals)
    assert np.isfinite(val)


def test_fit_marker_lrt_stable_for_near_singular_design() -> None:
    n = 12
    x = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    g = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.float64)
    y = 0.3 + 0.2 * x + 0.7 * g + np.array([0.03, -0.01, 0.02, -0.02, 0.01, -0.01, 0.02, -0.01, 0.01, -0.03, 0.02, -0.01])
    X = np.column_stack([np.ones(n), x, x + 1e-10])
    eigenvals = np.geomspace(1e-6, 3.0, n)

    null_neg_loglik = _calculate_neg_ml_likelihood(0.5, y, X, eigenvals)
    lrt_stat, p_value, beta, se = fit_marker_lrt(y, X, g, eigenvals, null_neg_loglik)

    assert np.isfinite(lrt_stat)
    assert 0.0 <= p_value <= 1.0
    assert np.isfinite(beta)
    assert np.isfinite(se)


def test_fit_marker_lrt_gemma_solver_matches_brent() -> None:
    rng = np.random.default_rng(42)
    n = 40
    x = rng.normal(size=n)
    g = rng.integers(0, 3, size=n).astype(np.float64)
    y = 0.5 + 0.6 * x + 0.4 * g + rng.normal(scale=0.3, size=n)
    X = np.column_stack([np.ones(n), x]).astype(np.float64)
    eigenvals = np.linspace(0.2, 2.0, n).astype(np.float64)

    null_neg_loglik = _calculate_neg_ml_likelihood(0.5, y, X, eigenvals)
    _, p_brent, beta_brent, se_brent = fit_marker_lrt(
        y, X, g, eigenvals, null_neg_loglik, solver="BRENT"
    )
    _, p_gemma, beta_gemma, se_gemma = fit_marker_lrt(
        y, X, g, eigenvals, null_neg_loglik, solver="GEMMA"
    )

    assert np.isfinite(p_brent) and np.isfinite(p_gemma)
    assert np.isfinite(beta_brent) and np.isfinite(beta_gemma)
    assert np.isfinite(se_brent) and np.isfinite(se_gemma)
    assert abs(-np.log10(max(p_brent, 1e-300)) + np.log10(max(p_gemma, 1e-300))) < 0.05
    assert abs(beta_brent - beta_gemma) < 0.02


def test_fit_marker_lrt_prebuilt_matches_standard() -> None:
    rng = np.random.default_rng(7)
    n = 36
    x = rng.normal(size=n)
    g = rng.integers(0, 3, size=n).astype(np.float64)
    y = 1.0 + 0.5 * x + 0.3 * g + rng.normal(scale=0.25, size=n)
    X = np.column_stack([np.ones(n), x]).astype(np.float64)
    X_alt = np.column_stack([X, g]).astype(np.float64)
    eigenvals = np.linspace(0.15, 1.8, n).astype(np.float64)

    null_neg_loglik = _calculate_neg_ml_likelihood(0.5, y, X, eigenvals)
    res_standard = fit_marker_lrt(y, X, g, eigenvals, null_neg_loglik, solver="GEMMA")
    res_prebuilt = fit_marker_lrt_prebuilt(
        y,
        X_alt,
        eigenvals,
        null_neg_loglik,
        solver_norm="GEMMA",
        assume_sanitized=True,
    )

    assert np.isfinite(res_standard[1]) and np.isfinite(res_prebuilt[1])
    assert np.isfinite(res_standard[2]) and np.isfinite(res_prebuilt[2])
    assert np.isfinite(res_standard[3]) and np.isfinite(res_prebuilt[3])
    assert abs(-np.log10(max(res_standard[1], 1e-300)) + np.log10(max(res_prebuilt[1], 1e-300))) < 1e-7
    assert abs(res_standard[2] - res_prebuilt[2]) < 1e-7
    assert abs(res_standard[3] - res_prebuilt[3]) < 1e-7


def test_fit_marker_lrt_gemma_stable_on_pathological_inputs() -> None:
    rng = np.random.default_rng(1234)
    n = 64
    x = rng.normal(size=n)
    g = np.zeros(n, dtype=np.float64)
    g[0] = 2.0  # Quasi-monomorphic marker to stress conditioning.
    y = 0.2 + 0.1 * x + 0.05 * g + rng.normal(scale=0.5, size=n)
    X = np.column_stack([np.ones(n), x]).astype(np.float64)
    # Weak kinship signal (eigenvalues clustered near 1) can push h2 to boundaries.
    eigenvals = (1.0 + np.linspace(-1e-8, 1e-8, n)).astype(np.float64)

    null_neg_loglik = _calculate_neg_ml_likelihood(0.5, y, X, eigenvals)
    _, p_brent, beta_brent, se_brent = fit_marker_lrt(
        y, X, g, eigenvals, null_neg_loglik, solver="BRENT"
    )
    _, p_gemma, beta_gemma, se_gemma = fit_marker_lrt(
        y, X, g, eigenvals, null_neg_loglik, solver="GEMMA"
    )

    assert 0.0 <= p_brent <= 1.0
    assert 0.0 <= p_gemma <= 1.0
    assert np.isfinite(beta_brent) and np.isfinite(beta_gemma)
    assert not np.isnan(se_brent) and not np.isnan(se_gemma)
    assert abs(-np.log10(max(p_brent, 1e-300)) + np.log10(max(p_gemma, 1e-300))) < 0.2
    assert abs(beta_brent - beta_gemma) < 0.1


def test_gauss_solve_flags_near_singular_design() -> None:
    # Directly exercise the GEMMA Gaussian-elimination kernel on the augmented
    # Gram matrix produced by a near-constant (effectively monomorphic, very
    # low-MAF) marker. Such a system is near-singular: with the old absolute
    # 1e-300 pivot threshold the elimination ran to completion and returned a
    # huge finite garbage inv_b22 (~1e+13 -> SE ~1e+6) instead of being flagged
    # singular. The relative pivot tolerance must now flag it (success=False)
    # so the caller routes to the exact lstsq fallback, matching how an exactly
    # singular system is handled.
    from panicle.association.lrt import HAS_NUMBA, _gauss_solve

    if not HAS_NUMBA:
        import pytest

        pytest.skip("Gaussian-elimination kernel only used on the Numba path")

    n = 200
    rng = np.random.default_rng(20260622)
    x = rng.normal(size=n)
    g = np.ones(n, dtype=np.float64)
    g[0] = 1.0 + 1e-9  # near-constant marker column -> near-singular design
    Xa = np.column_stack([np.ones(n), x, g]).astype(np.float64)
    inv = np.ones(n, dtype=np.float64)
    A = (Xa.T * inv) @ Xa  # Xa^T diag(inv) Xa, as built in _lrt_eval

    # inv_b22 is the last diagonal entry of A^-1: solve A z = e_last.
    e_last = np.zeros(3, dtype=np.float64)
    e_last[-1] = 1.0
    ok, z = _gauss_solve(A.copy(), e_last.copy())

    # Must be flagged singular and routed to fallback, never returning the
    # ~1e+13 finite garbage the absolute threshold produced.
    assert ok is False or (np.isfinite(z[-1]) and abs(z[-1]) < 1e6)

    # A well-conditioned design must still be solved (not over-flagged).
    g_good = rng.integers(0, 3, size=n).astype(np.float64)
    Xa_good = np.column_stack([np.ones(n), x, g_good]).astype(np.float64)
    A_good = (Xa_good.T * inv) @ Xa_good
    rhs = Xa_good.T @ (inv * rng.normal(size=n))
    ok_good, beta = _gauss_solve(A_good.copy(), rhs.copy())
    assert ok_good
    np.testing.assert_allclose(beta, np.linalg.solve(A_good, rhs), rtol=1e-8, atol=1e-10)


def test_fit_marker_lrt_auto_solver_uses_brent_for_small_n() -> None:
    rng = np.random.default_rng(99)
    n = 120
    x = rng.normal(size=n)
    g = rng.integers(0, 3, size=n).astype(np.float64)
    y = 0.6 + 0.5 * x + 0.25 * g + rng.normal(scale=0.4, size=n)
    X = np.column_stack([np.ones(n), x]).astype(np.float64)
    eigenvals = np.linspace(0.25, 1.9, n).astype(np.float64)

    null_neg_loglik = _calculate_neg_ml_likelihood(0.5, y, X, eigenvals)
    _, p_brent, beta_brent, se_brent = fit_marker_lrt(
        y, X, g, eigenvals, null_neg_loglik, solver="BRENT"
    )
    _, p_auto, beta_auto, se_auto = fit_marker_lrt(
        y, X, g, eigenvals, null_neg_loglik, solver="AUTO"
    )

    assert np.isfinite(p_auto) and np.isfinite(p_brent)
    assert np.isfinite(beta_auto) and np.isfinite(beta_brent)
    assert np.isfinite(se_auto) and np.isfinite(se_brent)
    assert abs(-np.log10(max(p_auto, 1e-300)) + np.log10(max(p_brent, 1e-300))) < 1e-9
    assert abs(beta_auto - beta_brent) < 1e-10
    assert abs(se_auto - se_brent) < 1e-10


def test_fit_markers_lrt_batch_prebuilt_matches_markerwise_gemma() -> None:
    rng = np.random.default_rng(2026)
    n = 96
    m = 48
    x = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x]).astype(np.float64)
    G = rng.integers(0, 3, size=(n, m)).astype(np.float64)
    y = 0.7 + 0.4 * x + 0.15 * G[:, 3] - 0.12 * G[:, 17] + rng.normal(scale=0.6, size=n)
    eigenvals = np.linspace(0.2, 2.4, n).astype(np.float64)
    null_h2 = 0.45
    null_neg_loglik = _calculate_neg_ml_likelihood(null_h2, y, X, eigenvals)

    p_batch, beta_batch, se_batch = fit_markers_lrt_batch_prebuilt(
        y,
        X,
        G,
        eigenvals,
        null_neg_loglik,
        null_h2=null_h2,
        solver_norm="GEMMA",
        assume_sanitized=True,
    )

    p_ref = np.empty(m, dtype=np.float64)
    beta_ref = np.empty(m, dtype=np.float64)
    se_ref = np.empty(m, dtype=np.float64)
    X_alt = np.empty((n, X.shape[1] + 1), dtype=np.float64)
    X_alt[:, :-1] = X
    for j in range(m):
        X_alt[:, -1] = G[:, j]
        _, p_val, beta_val, se_val = fit_marker_lrt_prebuilt(
            y,
            X_alt,
            eigenvals,
            null_neg_loglik,
            null_h2=null_h2,
            solver_norm="GEMMA",
            assume_sanitized=True,
        )
        p_ref[j] = p_val
        beta_ref[j] = beta_val
        se_ref[j] = se_val

    assert np.all(np.isfinite(p_batch))
    assert np.all(np.isfinite(beta_batch))
    assert np.all(np.isfinite(se_batch))

    logp_batch = -np.log10(np.clip(p_batch, 1e-300, 1.0))
    logp_ref = -np.log10(np.clip(p_ref, 1e-300, 1.0))
    assert np.max(np.abs(logp_batch - logp_ref)) < 0.05
    assert np.max(np.abs(beta_batch - beta_ref)) < 0.02
    assert np.max(np.abs(se_batch - se_ref)) < 0.02


def test_lrt_gemma_paths_suppress_matmul_runtime_warnings() -> None:
    n = 16
    m = 12
    scale = 1e200
    x = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    X = np.column_stack([np.ones(n), x]).astype(np.float64) * scale
    G = (
        np.tile(np.array([0.0, 1.0, 2.0, 1.0], dtype=np.float64), (n * m // 4) + 1)[: n * m]
        .reshape(n, m)
        * scale
    )
    y = (0.1 + (0.2 * x)) * scale
    eigenvals = np.linspace(0.2, 2.0, n, dtype=np.float64)

    with warnings.catch_warnings(record=True) as batch_caught:
        warnings.simplefilter("always", RuntimeWarning)
        p_batch, beta_batch, se_batch = fit_markers_lrt_batch_prebuilt(
            y,
            X,
            G,
            eigenvals,
            0.0,
            null_h2=0.5,
            solver_norm="GEMMA",
            assume_sanitized=False,
        )

    batch_matmul_warnings = [w for w in batch_caught if "encountered in matmul" in str(w.message)]
    assert not batch_matmul_warnings
    assert np.all(np.isfinite(p_batch))
    assert np.all(np.isfinite(beta_batch))
    assert np.all(np.isfinite(se_batch))

    X_alt = np.column_stack([X, G[:, 0]])
    with warnings.catch_warnings(record=True) as single_caught:
        warnings.simplefilter("always", RuntimeWarning)
        lrt_stat, p_value, beta, se = fit_marker_lrt_prebuilt(
            y,
            X_alt,
            eigenvals,
            0.0,
            null_h2=0.5,
            solver_norm="GEMMA",
            assume_sanitized=False,
        )

    single_matmul_warnings = [w for w in single_caught if "encountered in matmul" in str(w.message)]
    assert not single_matmul_warnings
    assert np.isfinite(lrt_stat)
    assert 0.0 <= p_value <= 1.0
    assert np.isfinite(beta)
    assert np.isfinite(se)


def test_fit_markers_lrt_batch_two_bracket_markers_stay_in_kernel(monkeypatch) -> None:
    """Markers whose profile-ML derivative changes sign on both sides of the
    null h2 (pattern -,+,- at h2_min / null h2 / h2_max) must be resolved inside
    the compiled kernel and agree with the scalar exact solver.

    A centered kinship has one ~zero eigenvalue along the intercept direction;
    its log-det term makes the derivative negative at h2_max, so every marker
    whose alternative h2 falls below the null h2 shows two brackets."""
    import panicle.association.lrt as lrt_mod

    if not lrt_mod.HAS_NUMBA:
        return
    rng = np.random.default_rng(7)
    n, m = 200, 64
    Z = rng.normal(size=(n, 400))
    Z -= Z.mean(axis=0)
    K = Z @ Z.T / 400
    evals, U = np.linalg.eigh(K)
    evals = np.maximum(evals, 1e-6)
    G_raw = rng.integers(0, 3, size=(n, m)).astype(np.float64)
    u = rng.multivariate_normal(np.zeros(n), K + 1e-8 * np.eye(n))
    y_raw = 1.0 + 0.8 * u + G_raw[:, :8] @ np.full(8, 0.5) + rng.normal(scale=0.7, size=n)
    y = U.T @ y_raw
    X = U.T @ np.ones((n, 1))
    G = U.T @ G_raw

    grid = np.linspace(0.01, 0.99, 99)
    nll = [_calculate_neg_ml_likelihood(h, y, X, evals) for h in grid]
    null_h2 = float(grid[int(np.argmin(nll))])
    null_neg = float(min(nll))

    X_alt = np.empty((n, 2))
    X_alt[:, :-1] = X
    ref = []
    for j in range(m):
        X_alt[:, -1] = G[:, j]
        ref.append(fit_marker_lrt_prebuilt(
            y, X_alt, evals, null_neg, null_h2=null_h2,
            solver_norm="GEMMA", assume_sanitized=True)[1:])
    ref = np.asarray(ref)

    # Confirm the scenario really contains two-bracket markers.
    lam_min = lrt_mod._H2_MIN / (1 - lrt_mod._H2_MIN)
    lam_max = lrt_mod._H2_MAX / (1 - lrt_mod._H2_MAX)
    md = evals - 1.0
    n_two = 0
    for j in range(m):
        X_alt[:, -1] = G[:, j]
        g = [lrt_mod._lrt_eval(h, y, X_alt, evals, md, float(n), False)[1]
             for h in (lrt_mod._H2_MIN, null_h2, lrt_mod._H2_MAX)]
        n_two += int(g[0] < 0 < g[1] and g[2] < 0)
    assert n_two > 0

    calls = {"n": 0}
    orig = lrt_mod._fit_marker_lrt_core

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(lrt_mod, "_fit_marker_lrt_core", counting)
    p, b, s = fit_markers_lrt_batch_prebuilt(
        y, X, G, evals, null_neg, null_h2=null_h2,
        solver_norm="GEMMA", assume_sanitized=True)
    assert calls["n"] == 0
    logp = -np.log10(np.clip(p, 1e-300, 1.0))
    logp_ref = -np.log10(np.clip(ref[:, 0], 1e-300, 1.0))
    assert np.max(np.abs(logp - logp_ref)) < 1e-6
    # Root tolerance (1e-4 relative in lambda) bounds beta/se agreement.
    assert np.max(np.abs(b - ref[:, 1])) < 1e-4
    assert np.max(np.abs(s - ref[:, 2])) < 1e-4
