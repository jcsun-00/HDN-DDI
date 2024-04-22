#!/bin/bash

# Shell脚本：重复循环，每次循环保持等待，日志文件中包含"Best Result"时开始下次循环
for ((k=2; k<=3; k++)); do
    for ((i=0; i<=4; i++)); do
        echo "$i-th running"

        f=$i # fold
        comment="HDNDDI_fold$f"
        time=$(date +%m%d_%H%M)
        log_name=$time\_$comment.log
        log_path=log/new_pkl/$log_name
        nohup python drugbank_test/inductive_train.py \
            --fold $f \
            --batch_size 1024 \
            --n_atom_feats 66 \
            --device 0 \
            > $log_path & \

        # 启动tail -f命令并将其放入后台
        tail -f $log_path &
        tail_pid=$!

        # 检查test.log文件中是否包含“Best Result”
        sleep 30
        while ! grep -q "Result" $log_path; do
            sleep 30  # 60秒检查一次
        done

        kill -15 $tail_pid  # 终止tail

    done
done
