import fire
import jsonlines
from pathlib import Path


def save(out_dirpath, split, lines):
    out_dirpath.mkdir(parents=True, exist_ok=True)
    filepath = out_dirpath / f"{split}.jsonl"
    with jsonlines.open(filepath, "w") as f:
        f.write_all(lines)
    print(f"Saved {len(lines)} lines to {filepath}")


def read_file(filepath):
    with jsonlines.open(filepath) as f:
        lines = [line for line in f]
    return lines


def main(dirpath: str):
    dirpath = Path(dirpath)
    splits = ["train", "dev", "test"]
    langs = ["en", "es", "fr", "id", "ja", "zh"]
    for split in splits:
        lines_final = []
        for lang in langs:
            lines_final.extend(read_file(dirpath / lang / f"{split}.jsonl"))
        save(dirpath / "mixed", split, lines_final)


if __name__ == "__main__":
    fire.Fire(main)
