from os.path import abspath, dirname, join
import argparse
import json
import sys
import numpy as np

ROOT_DIR = dirname(dirname(abspath(__file__)))
print("Root directory: ", ROOT_DIR)
sys.path.append(ROOT_DIR)

from scm_ds.scm import *


ST_X_LOWER = -5.0
ST_X_UPPER = 5.0
ST_X_STAR = -2.903534


def _build_styblinski_tang_dataset(dim: int) -> SCMDataset:
    p_names = [f"P{i+1}" for i in range(dim)]
    sty_terms = [f"(P{i+1}**4 - 16*P{i+1}**2 + 5*P{i+1})" for i in range(dim)]
    y_expr = "0.5*(" + " + ".join(sty_terms) + ") + eps_Y"
    specs = [NodeSpec(name, [], f"eps_{name}") for name in p_names]
    specs.append(NodeSpec("Y", p_names, y_expr))

    singles = {name: (lambda rng, n: rng.uniform(ST_X_LOWER, ST_X_UPPER, size=n)) for name in p_names}
    singles["Y"] = lambda rng, n: np.zeros(n)

    return SCMDataset(
        name=f"styblinski_tang_{dim}d",
        description=(
            "d-dimensional Styblinski-Tang with x sampled in [-5,5]^d; "
            "inputs normalized to [-1,1]^d and outputs standardized after sampling."
        ),
        tags=None,
        specs=specs,
        params={},
        singles=singles,
        groups=None,
        input_labels=p_names,
        target_labels=["Y"],
    )


def _styblinski_tang_raw(x_raw: np.ndarray) -> np.ndarray:
    # x_raw shape: [N, d]
    return 0.5 * np.sum(x_raw**4 - 16.0 * x_raw**2 + 5.0 * x_raw, axis=1)


def _postprocess_and_save_stats(
    save_dir: str,
    dim: int,
    n_local: int = 0,
    local_radius: float = 0.25,
    local_seed: int = 42,
) -> None:
    ds_path = join(save_dir, "ds.npz")
    loaded = np.load(ds_path)
    x_arr = loaded["x"].copy()
    y_arr = loaded["y"].copy()

    # Optional local enrichment around the theoretical minimum x*=(-2.903534,...,-2.903534).
    if n_local > 0:
        if local_radius <= 0.0:
            raise ValueError("--local-radius must be > 0 when --n-local > 0")
        rng = np.random.default_rng(local_seed)
        x_local_raw = np.clip(
            ST_X_STAR + rng.uniform(-local_radius, local_radius, size=(n_local, dim)),
            ST_X_LOWER,
            ST_X_UPPER,
        )
        y_local_raw = _styblinski_tang_raw(x_local_raw)

        x_local = np.zeros((n_local, dim, 2), dtype=x_arr.dtype)
        x_local[:, :, 0] = x_local_raw.astype(x_arr.dtype, copy=False)
        # keep variable-id channel consistent with existing dataset encoding
        x_local[:, :, 1] = np.asarray(x_arr[0, :, 1], dtype=x_arr.dtype)[None, :]

        y_local = np.zeros((n_local, y_arr.shape[1], y_arr.shape[2]), dtype=y_arr.dtype)
        y_local[:, 0, 0] = y_local_raw.astype(y_arr.dtype, copy=False)
        y_local[:, 0, 1] = np.asarray(y_arr[0, 0, 1], dtype=y_arr.dtype)

        x_arr = np.concatenate([x_arr, x_local], axis=0)
        y_arr = np.concatenate([y_arr, y_local], axis=0)

    # Normalize x from [-5,5] to [-1,1]
    x_vals = x_arr[:, :, 0]
    x_arr[:, :, 0] = 2.0 * ((x_vals - ST_X_LOWER) / (ST_X_UPPER - ST_X_LOWER)) - 1.0

    # Standardize y
    y_vals = y_arr[:, 0, 0]
    y_mean = float(np.mean(y_vals))
    y_std = float(np.std(y_vals))
    if y_std <= 0.0:
        y_std = 1.0
    y_arr[:, 0, 0] = (y_vals - y_mean) / y_std

    np.savez_compressed(ds_path, x=x_arr, y=y_arr)

    # Persist stats in meta.json so optimization/evaluation can de-normalize
    meta_path = join(save_dir, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["function"] = "styblinski_tang"
    meta["x_lower"] = ST_X_LOWER
    meta["x_upper"] = ST_X_UPPER
    meta["y_mean"] = y_mean
    meta["y_std"] = y_std
    meta["local_sampling"] = {
        "enabled": bool(n_local > 0),
        "n_local": int(n_local),
        "local_radius": float(local_radius),
        "local_seed": int(local_seed),
        "x_star": float(ST_X_STAR),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Styblinski-Tang dataset at data/example.")
    parser.add_argument("--dim", type=int, default=20, help="Problem dimension d.")
    parser.add_argument("--n", type=int, default=50000, help="Number of sampled points.")
    parser.add_argument(
        "--n-local",
        type=int,
        default=0,
        help="Additional points sampled locally around theoretical minimum x* (raw x-space).",
    )
    parser.add_argument(
        "--local-radius",
        type=float,
        default=0.25,
        help="Half-width around x* for local uniform sampling in raw x-space.",
    )
    parser.add_argument("--local-seed", type=int, default=42, help="RNG seed for local sampling.")
    args = parser.parse_args()

    ds_scm = _build_styblinski_tang_dataset(args.dim)
    save_dir = join(ROOT_DIR, "data/example")
    ds_scm.generate_ds(mode="flat", n=args.n, save_dir=save_dir)
    _postprocess_and_save_stats(
        save_dir=save_dir,
        dim=args.dim,
        n_local=args.n_local,
        local_radius=args.local_radius,
        local_seed=args.local_seed,
    )


# Keep compatibility with existing imports in optimization scripts.
ds_scm_quad = _build_styblinski_tang_dataset(dim=20)


if __name__ == "__main__":
    main()
