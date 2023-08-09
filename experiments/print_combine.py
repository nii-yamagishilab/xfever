from print_mono_multi import LANGS, get_acc_ece, avg

MAPP = {
    "-": "--",
    "js": "Pred ($\\JS$)",
    "mse-feat": "Repr ($\\MSE$-feat)",
    "js+mse-feat": "Pred ($\\JS$) \\& Pepr ($\\MSE$-feat)",
}


def print_row(reg):
    to_keep = []
    train = "para"
    pretrained_lrs = [
        ("bert-base-multilingual-cased", "2e-5"),
        ("xlm-roberta-large", "5e-6"),
    ]
    for pretrained, lr in pretrained_lrs:
        accs = []
        for lang in LANGS:
            filename = (
                f"{train}_{pretrained}_{lr}/{pretrained}-128-{lang}-out/eval.test.txt"
                if reg == "-"
                else f"{train}_{pretrained}_{lr}+{reg}/{pretrained}-128-{lang}-out/eval.test.txt"
            )
            acc, _ = get_acc_ece(filename)
            accs.append(acc)
        to_keep.append(avg(accs))

    res = " & ".join(map(str, to_keep))
    print(f"{MAPP[reg]} & {res} \\\\")


def main():
    print("\\toprule")
    header = " & ".join(["Consistency ($\\reg$)", "mBERT", "XLM-R-large"])
    print(f"{header} \\\\")
    print("\\midrule")
    print_row("-")
    print_row("js")
    print_row("mse-feat")
    print_row("js+mse-feat")
    print("\\bottomrule")


if __name__ == "__main__":
    main()
