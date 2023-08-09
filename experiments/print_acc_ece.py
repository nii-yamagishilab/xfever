import fire
from print_mono_multi import LANGS, get_acc_ece, avg

MAPP = {
    "en": "Zero-shot $\\avgLossZ$     & -- & --",
    "mixed": "Non-parallel $\\avgLossNP$ & -- & --",
    "para": "Parallel $\\avgLossP$      & -- & --",
    "para kl": "                     & Pred & $\\KL$",
    "para j": "                     &      & $\\J$",
    "para j-0.25": "                     &      & $\\J$",
    "para js": "                     &      & $\\JS$",
    "para mse-feat": "                     & Repr & $\\MSE$-feat",
    "para mse-penu": "                     &      & $\\MSE$-penu",
    "para cos-feat": "                     &      & $\\COS$-feat",
    "para cos-penu": "                     &      & $\\COS$-penu",
}


def print_row(
    score,
    pretrained,
    lr,
    train,
    cst=None,
):
    to_keep = []
    for lang in LANGS:
        if cst is None:
            filename = (
                f"{train}_{pretrained}_{lr}/{pretrained}-128-{lang}-out/eval.test.txt"
            )
        else:
            filename = f"{train}_{pretrained}_{lr}+{cst}/{pretrained}-128-{lang}-out/eval.test.txt"
        acc, ece = get_acc_ece(filename)
        if score == "acc":
            to_keep.append(acc)
        elif score == "ece":
            to_keep.append(ece)
        else:
            raise KeyError(score)
    to_keep.append(avg(to_keep))
    res = " & ".join(map(str, to_keep))
    if cst is None:
        key = f"{train}"
    else:
        key = f"{train} {cst}"
    print(f"{MAPP[key]} & {res} \\\\")


def main(pretrained: str, score: str):
    if pretrained == "bert-base-multilingual-cased":
        lr = "2e-5"
        j = "j"
    elif pretrained == "xlm-roberta-large":
        lr = "5e-6"
        j = "j-0.25"
    else:
        raise KeyError(pretrained)

    print("\\toprule")
    header = " & ".join(LANGS + ["Avg"])
    print(f"Model & Consistency & $\\reg$ & {header} \\\\")
    print("\\midrule")
    print_row(score, pretrained, lr, "en", None)
    print_row(score, pretrained, lr, "mixed", None)
    print_row(score, pretrained, lr, "para", None)
    print_row(score, pretrained, lr, "para", "kl")
    print_row(score, pretrained, lr, "para", j)
    print_row(score, pretrained, lr, "para", "js")
    print_row(score, pretrained, lr, "para", "mse-feat")
    print_row(score, pretrained, lr, "para", "mse-penu")
    print_row(score, pretrained, lr, "para", "cos-feat")
    print_row(score, pretrained, lr, "para", "cos-penu")
    print("\\bottomrule")


if __name__ == "__main__":
    fire.Fire(main)
