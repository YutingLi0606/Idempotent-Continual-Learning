for N_TASKS in  10
do
    for SEED in 0 1 2 3 4
    do

 
   
    ASCEND_VISIBLE_DEVICES=0 python main.py --model="ider" --load_best_args --savecheckpoint=True  --class_balance=True  --dataset="seq-tinyimg" --device="npu:0" --seed=$SEED --n_tasks=$N_TASKS --buffer_size=4000 --run_name=" er+id seed $SEED $N_TASKS tasks" --experiment_name="idempotent/tinyimg/buffer4000"

    done
done