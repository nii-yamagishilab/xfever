from pathlib import Path

MAPP = {
    "bert-base-uncased": "BERT",
    "roberta-base": "RoBERTa-base",
    "roberta-large": "RoBERTa-large",
    "bert-base-multilingual-cased": "mBERT",
    "xlm-roberta-base": "XLM-R-base",
    "xlm-roberta-large": "XLM-R-large",
}

LANGS = ["en", "es", "fr", "id", "ja", "zh"]


def get_acc_ece(fname):
    filepath = Path(fname)
    assert filepath.exists(), filepath
    lines = [line.strip() for line in open(filepath, "r")]
    acc, ece = lines[-2].lower(), lines[-1].lower()
    assert "acc: " == acc[:5] and "ece: " == ece[:5]
    return float(acc[5:]), float(ece[5:])


def avg(lst):
    return round(sum(lst) / len(lst), 1)


def print_row(train, pretrained, lr):
    to_keep = []
    for lang in LANGS:
        filename = (
            f"{train}_{pretrained}_{lr}/{pretrained}-128-{lang}-out/eval.test.txt"
        )
        acc, _ = get_acc_ece(filename)
        to_keep.append(acc)
    to_keep.append(avg(to_keep))
    res = " & ".join(map(str, to_keep))
    print(f"{MAPP[pretrained]} & {res} \\\\")


def main():
    print("\\toprule")
    header = " & ".join(LANGS + ["Avg"])
    print(f"$\\PLM$ & {header} \\\\")
    print("\\midrule")
    print("{\\em Monolingual} \\\\")
    print_row("en", "bert-base-uncased", "2e-5")
    print_row("en", "roberta-base", "2e-5")
    print_row("en", "roberta-large", "5e-6")
    print("\\midrule")
    print("{\\em Multilingual} \\\\")
    print_row("en", "bert-base-multilingual-cased", "2e-5")
    print_row("en", "xlm-roberta-base", "2e-5")
    print_row("en", "xlm-roberta-large", "5e-6")
    print("\\bottomrule")


if __name__ == "__main__":
    main()
