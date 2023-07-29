from print_tab_3 import LANGS, get_acc_ece, avg

MAPP = {
    "en": "Zero-shot $\\avgLossZ$     & -- & --",
    "mixed": "Non-parallel $\\avgLossNP$ & -- & --",
    "para kl": "Parallel $\\avgLossP$ & Pred & $\\KL$",
    "para j": "                     &      & $\\J$",
    "para js": "                     &      & $\\JS$",
    "para mse-feat": "                     & Repr & $\\MSE$-feat",
    "para mse-penu": "                     &      & $\\MSE$-penu",
    "para cos-feat": "                     &      & $\\COS$-feat",
    "para cos-penu": "                     &      & $\\COS$-penu",
}


def print_row(
    train,
    cst=None,
    pretrained="bert-base-multilingual-cased",
    lr="2e-5",
):
    to_keep = []
    for lang in LANGS:
        if cst is None:
            filename = (
                f"{train}_{pretrained}_{lr}/{pretrained}-128-{lang}-out/eval.test.txt"
            )
        else:
            filename = f"{train}_{pretrained}_{lr}+{cst}/{pretrained}-128-{lang}-out/eval.test.txt"
        acc, _ = get_acc_ece(filename)
        to_keep.append(acc)
    to_keep.append(avg(to_keep))
    res = " & ".join(map(str, to_keep))
    if cst is None:
        key = f"{train}"
    else:
        key = f"{train} {cst}"
    print(f"{MAPP[key]} & {res} \\\\")


def main():
    print("\\toprule")
    header = " & ".join(LANGS + ["Avg"])
    print(f"Model & Consistency & $\\reg$ & {header} \\\\")
    print("\\midrule")
    print_row("en")
    print_row("mixed")
    print_row("para", "kl")
    print_row("para", "j")
    print_row("para", "js")
    print_row("para", "mse-feat")
    print_row("para", "mse-penu")
    print_row("para", "cos-feat")
    print_row("para", "cos-penu")
    print("\\bottomrule")


if __name__ == "__main__":
    main()
