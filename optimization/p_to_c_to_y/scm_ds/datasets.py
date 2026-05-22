from os.path import abspath, dirname
import sys

ROOT_DIR = dirname(dirname(abspath(__file__)))
print("Root directory: ", ROOT_DIR)
sys.path.append(ROOT_DIR)

from scm_ds.scm import *



ds_scm_quad = SCMDataset(
    name="quadratic_two_parent",
    description="Y = P1^2 + P2^2",
    tags=None,
    specs=[
        NodeSpec("P1", [], "eps_P1"),
        NodeSpec("P2", [], "eps_P2"),
        NodeSpec("C1", ["P1"], "P1**2 + eps_C1"),
        NodeSpec("C2", ["P2"], "P2**2 + eps_C2"),
        NodeSpec("Y", ["C1", "C2"], "C1 + C2 + eps_Y"),
    ],
    params={
        "w1": 0.01,
        "w2": 0.01,
        # or any params your SCM uses
    },
    singles={
        "P1": lambda rng,n: rng.uniform(-1, 1, size=n),
        "P2": lambda rng,n: rng.uniform(-1, 1, size=n),
        "C1": lambda rng,n: 0.1 * rng.standard_normal(n),
        "C2": lambda rng,n: 0.1 * rng.standard_normal(n),
        "Y": lambda rng,n: 0.1 * rng.standard_normal(n),
    },
    groups=None,
    input_labels=["P1","P2","C1","C2"],
    target_labels=["Y"],
) 
ds_scm_quad.generate_ds(mode="flat", n=6000, save_dir=join(ROOT_DIR,"data/example"))

    

# TODO
# - one-to-one-CT
# - one-to-many-noCT
# - one-to-many-noCT



