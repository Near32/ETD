get_free_gpu() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{print NR-1 " " $1}' | sort -k2 -n -r | awk '{print $1}' | head -n 1
}

session=MiniWorld

# 检查 session 是否存在，如果不存在则创建
tmux has-session -t $session 2>/dev/null || tmux new-session -d -s $session

envs=("FullMazeS5")
seeds=(10 11 12 13 14)
# [NoModel|DEIR|ICM|RND|NGU|NovelD|PlainDiscriminator|PlainInverse|PlainForward]
mtd="TDD"

# 循环创建 tmux 窗口并运行 Python 程序
for env in "${envs[@]}"
do
    for seed in "${seeds[@]}"
    do
        exp_name="${mtd}_p16_s512_b512_d512_int1e-2_mlr1e-4"
        tmux select-window -t "$session":"$exp_name-$env-$seed" || tmux new-window -n "$exp_name-$env-$seed" -t $session
        # Get the ID of the most free GPU
        free_gpu=$(get_free_gpu)

        tmux send-keys -t "$session":"$exp_name-$env-$seed" "conda activate tdd" C-m
        tmux send-keys -t "$session":"$exp_name-$env-$seed" "CUDA_VISIBLE_DEVICES=$free_gpu PYTHONPATH='./' \
        xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' python3 src/train.py \
        --game_name=$env \
        --run_id=$seed \
        --int_rew_source=${mtd} \
        --env_source=miniworld \
        --exp_name=$exp_name \
        --use_wandb=1 \
        --int_rew_coef=1e-2 \
        --ext_rew_coef=1.0 \
        --model_learning_rate=1e-4 \
        --ent_coef=1e-2 \
        --max_grad_norm=0.5 \
        --tdd_aggregate_func=min \
        --model_n_epochs=4 \
        --use_model_rnn=0 \
        --record_video=0 \
        --enable_plotting=0 \
        --model_mlp_layers=1 \
        --total_steps=5_000_000 \
        --image_noise_scale=0 \
        --batch_size=512 \
        --discount=0.99 \
        --policy_cnn_type=1 \
        --model_cnn_type=1 \
        --n_steps=512 \
        --num_processes=16 \
        --features_dim=512 \
        --model_features_dim=512 \
        --model_cnn_norm=LayerNorm \
        --model_mlp_norm=LayerNorm \
        --policy_cnn_norm=LayerNorm \
        --policy_mlp_norm=LayerNorm \
        " C-m
        # tmux send-keys -t "$session":"$exp_name-$env" "echo success" C-m
        sleep 15
    done
done

tmux attach-session -d