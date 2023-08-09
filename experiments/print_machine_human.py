from print_mono_multi import get_acc_ece, avg

MAPP = {
    "en bert-base-multilingual-cased 6h": "Zero-shot $\\avgLossZ$ & mBERT & Machine",
    "en bert-base-multilingual-cased 6h.human": "                &             & Human  ",
    "en xlm-roberta-large 6h": "                & XLM-R-large & Machine",
    "en xlm-roberta-large 6h.human": "                &             & Human  ",
    "mixed bert-base-multilingual-cased 6h": "Translate-train $\\avgLossNP$ & mBERT & Machine",
    "mixed bert-base-multilingual-cased 6h.human": "                       &             & Human  ",
    "mixed xlm-roberta-large 6h": "                       & XLM-R-large & Machine",
    "mixed xlm-roberta-large 6h.human": "                       &             & Human  ",
}

LANGS = ["es", "fr", "id", "ja", "zh"]


def print_row(train, pretrained, lr, suffix):
    to_keep = []
    for lang in LANGS:
        filename = f"{train}_{pretrained}_{lr}/{pretrained}-128-{lang}-out/eval.test.{suffix}.txt"
        acc, _ = get_acc_ece(filename)
        to_keep.append(acc)
    to_keep.append(avg(to_keep))
    res = " & ".join(map(str, to_keep))
    key = f"{train} {pretrained} {suffix}"
    print(f"{MAPP[key]} & {res} \\\\")


def main():
    print("\\toprule")
    header = " & ".join(LANGS + ["Avg"])
    print(f"Scenario & $\\PLM$ & Trans & {header} \\\\")
    print("\\midrule")
    print_row("en", "bert-base-multilingual-cased", "2e-5", "6h")
    print_row("en", "bert-base-multilingual-cased", "2e-5", "6h.human")
    print_row("en", "xlm-roberta-large", "5e-6", "6h")
    print_row("en", "xlm-roberta-large", "5e-6", "6h.human")
    print("\\midrule")
    print_row("mixed", "bert-base-multilingual-cased", "2e-5", "6h")
    print_row("mixed", "bert-base-multilingual-cased", "2e-5", "6h.human")
    print_row("mixed", "xlm-roberta-large", "5e-6", "6h")
    print_row("mixed", "xlm-roberta-large", "5e-6", "6h.human")
    print("\\bottomrule")


if __name__ == "__main__":
    main()
