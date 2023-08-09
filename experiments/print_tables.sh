#!/bin/bash

echo "% Table 3"
python print_mono_multi.py
echo "--"

echo "% Table 4"
python print_acc_ece.py bert-base-multilingual-cased acc
echo "--"

echo "% Table 5"
python print_combine.py
echo "--"

echo "% Table 6"
python print_acc_ece.py bert-base-multilingual-cased ece
echo "--"

echo "% Table 7"
python print_machine_human.py
echo "--"

echo "% Table 8"
python print_acc_ece.py xlm-roberta-large acc
echo "--"

echo "% Table 9"
python print_acc_ece.py xlm-roberta-large ece
echo "--"
