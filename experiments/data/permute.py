import fire
import jsonlines
import numpy as np
from pathlib import Path


def main(in_file: str, out_file: str, seed=3435):
    lines = [line for line in jsonlines.open(Path(in_file), "r")]
    rng = np.random.RandomState(seed)
    perm_idxs = rng.permutation(len(lines))
    perm_lines = [lines[i] for i in perm_idxs]
    with jsonlines.open(Path(out_file), "w") as out:
        out.write_all(perm_lines)


if __name__ == "__main__":
    fire.Fire(main)
