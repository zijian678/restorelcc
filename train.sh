
#nohup bash train.sh > train.log 2>&1 &

python RestoreLCC_train.py --spec_task alpaca --num_train_samples 2000 --lr 1e-3 --use_topk_heads 128
python RestoreLCC_train.py --spec_task alpaca --num_train_samples 2000 --lr 1e-3 --use_topk_heads 256
python RestoreLCC_train.py --spec_task alpaca --num_train_samples 2000 --lr 1e-3 --use_topk_heads 512


