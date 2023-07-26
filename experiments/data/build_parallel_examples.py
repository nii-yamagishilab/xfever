import copy
import fire
from pathlib import Path
from build_mixed_examples import read_file, save


def concat_inputs(lines_en, lines_lang):
    lines_concat = copy.deepcopy(lines_en)
    for line_en, line_lang, line_concat in zip(lines_en, lines_lang, lines_concat):
        assert line_en["id"] == line_lang["id"]
        line_concat["claim"] = line_en["claim"] + "[SEP]" + line_lang["claim"]
        line_concat["evidence"] = line_en["evidence"] + "[SEP]" + line_lang["evidence"]
    return lines_concat


def main(dirpath: str):
    dirpath = Path(dirpath)
    splits = ["train", "dev", "test"]
    langs = ["en", "es", "fr", "id", "ja", "zh"]
    for split in splits:
        lines_final = []
        lines_en = read_file(dirpath / langs[0] / f"{split}.jsonl")
        for lang in langs[1:]:
            lines_lang = read_file(dirpath / lang / f"{split}.jsonl")
            lines_final.extend(concat_inputs(lines_en, lines_lang))
        save(dirpath / "para", split, lines_final)


if __name__ == "__main__":
    fire.Fire(main)
