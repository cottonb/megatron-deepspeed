#!/bin/bash
# This example script is contributed by external user https://github.com/LydiaXiaohongLi





export https_proxy=http://10.10.20.100:1089 http_proxy=http://10.10.20.100:1089 all_proxy=socks5://10.10.20.100:1089
pip install sentencepiece
pip install datasets
pip install fire
pip install loguru
pip install sh
pip install py-cpuinfo
pip install hjson
pip install transformers
unset https_proxy http_proxy all_proxy

export PYTHONPATH=$PYTHONPATH:/mnt/huangyonghua/huangyonghua/DeepSpeed

export NCCL_IB_HCA=mlx5_2

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1} #PP

GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
# GPUS_PER_NODE=1
# Change for multinode config
# mizar 注入 MASTER_ADDR MASTER_PORT
MASTER_ADDR=${MASTER_ADDR:-localhost}
# echo $MASTER_ADDR
MASTER_PORT=9200

# mizar 注入的变量WORLD_SIZE代表节点数
NNODES=${WORLD_SIZE:-1}
# echo $NNODES
# mizar 注入rank
# NODE_RANK=$(echo $(hostname) | awk -F '-' '{print $NF}')
export NODE_RANK=${RANK:-0}


rm -rf /mnt/huangyonghua/huangyonghua/res/checkpoint_quant/1.3b_int_i1000/._dlckpt_ckpt_stage


export DLCKPT_RUN_ID="JOB_$NODE_RANK"
echo "节点rank: $NODE_RANK"
# echo $NODE_RANK
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
# WORLD_SIZE=1

export LOCAL_WORLD_SIZE=$GPUS_PER_NODE
echo 本地大小：$LOCAL_WORLD_SIZE
export BACKUP_MASTER_ADDR=$MASTER_ADDR
export BACKUP_MASTER_PORT=22230
export NODE_NUM=$NNODES
export IS_CHUNK_MODEL=1
export IS_CHUNK_OPTIM=0
export IS_BACKUP=1

# Parallel config
export TP=$GPUS_PER_NODE
# TP=2
export PP=1

# Data config
export DP_SIZE=$((WORLD_SIZE/TP/PP))
BATCH_SIZE=${BATCH_SIZE:-1}
# BATCH_SIZE=2
ACCUMULATION_STEPS=${ACCUMULATION_STEPS:-8}
GLOBAL_BATCH_SIZE=$(($BATCH_SIZE*$DP_SIZE*$ACCUMULATION_STEPS))

# Model config
SEQ_LENGTH=${SEQ_LENGTH:-4096}

# Original
HIDDEN_SIZE=512
FFN_HIDDEN_SIZE=2048
NUM_LAYERS=16
NUM_HEADS=16
NUM_ATTN_HEADS=16

# DATA_PATH=${DATA_PATH:-/mnt/resource/tmp_share/tanzheyue/data/long-instruction-clean-hf/256k}
DATA_PATH=${DATA_PATH:-/mnt/huangyonghua/huangyonghua/long-inst-4k}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-10000}
#TRAIN_ITERS=$(($TRAIN_SAMPLES/$GLOBAL_BATCH_SIZE))
TRAIN_ITERS=${TRAIN_ITERS:-5}

VPP_SIZE=$(($NUM_LAYERS/$TP))

EXP_NAME=${EXP_NAME:-profiling-gpt}
EXP_FOLDER=~/$EXP_NAME
export CHECKPOINT_PATH=/mnt/huangyonghua/huangyonghua/res/checkpoint_quant/1.3b_int_i1000
# TENSORBOARD_PATH=$EXP_FOLDER/tensorboard/llama_exp
TENSORBOARD_PATH=/mnt/huangyonghua/huangyonghua/Megatron-LM/tensorboard/llama_exp/1.3b_int8_i1000/no_quant

mkdir -p $CHECKPOINT_PATH $TENSORBOARD_PATH

TOKENIZER_PATH=${TOKENIZER_PATH:-/mnt/huangyonghua/huangyonghua/model/tokenizer.model}
TOKENIZER_TYPE=${TOKENIZER_TYPE:-Llama2Tokenizer}
# TOKENIZER_TYPE=${TOKENIZER_TYPE:-NullTokenizer}

SEED=${SEED:-42}

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT"

BACKUP_DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $BACKUP_PORT"

echo $DISTRIBUTED_ARGS

GPT_ARGS="
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --sequence-parallel \
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --num-attention-heads $NUM_ATTN_HEADS \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --seq-length $SEQ_LENGTH \
    --max-position-embeddings $SEQ_LENGTH \
    --position-embedding-type rope \
    --micro-batch-size $BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --vocab-size 55295\
    --make-vocab-size-divisible-by $TP \
    --lr 1e-5 \
    --lr-decay-iters $TRAIN_ITERS \
    --lr-decay-style cosine \
    --lr-warmup-fraction 0.003 \
    --min-lr 1.5e-6 \
    --normalization RMSNorm \
    --norm-epsilon 1e-5 \
    --weight-decay 1e-2 \
    --swiglu \
    --no-masked-softmax-fusion \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --clip-grad 1.0 \
    --bf16 \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 2 \
    --no-check-for-nan-in-loss-and-grad" 

DATA_ARGS="
    --data-path $DATA_PATH \
    --train-iters $TRAIN_ITERS \
    --tokenizer-type $TOKENIZER_TYPE \
    --tokenizer-model $TOKENIZER_PATH \
    --dataloader-type cyclic \
    --num-workers 8 \
    --split 1000,0,0"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval  1\
    --eval-interval 1000 \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --eval-iters 0"

#    --save $CHECKPOINT_PATH \
#    --load $CHECKPOINT_PATH \
    # --save $CHECKPOINT_PATH \
        # --tensorboard-dir $TENSORBOARD_PATH \
    # --save $CHECKPOINT_PATH \
# 用load前记得先不用load跑一遍，不然shm里没数据，第二次跑记得加上load

ZERO_STAGE=0
BASE_PATH=./tmp
DS_CONFIG=${BASE_PATH}/deepspeed.json

cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH_SIZE,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "steps_per_print": 1,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "bf16": {
    "enabled": true
  }
}
EOT

ds_args=""
ds_args=" --deepspeed ${ds_args}"
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"

if [ "${activation_checkpoint}" = "true" ]; then
  ds_args="--deepspeed-activation-checkpointing ${ds_args}"

  ## old argument for recomputing the transformer layer
  # ds_args="--checkpoint-activations ${ds_args}"

  ## new argument for recomputing the transformer layer
  ds_args="--recompute-granularity full --recompute-method uniform ${ds_args}"
  ## new argument for recomputing only the attention layer
  # ds_args="--recompute-granularity selective ${ds_args}"
fi

LOG_NAME="./dlckpt_results/output${NODE_RANK}.txt"
torchrun $DISTRIBUTED_ARGS ../pretrain_llama.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --seed $SEED \
    --distributed-backend nccl \
    --tensorboard-dir $TENSORBOARD_PATH \
    --use-flash-attn \
    --use-distributed-optimizer \
    --empty-unused-memory-level 2 \
    --tensorboard-log-interval 1 \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-memory-to-tensorboard \
    # 2>&1 | tee ${LOG_NAME} 

# torchrun $BACKUP_DISTRIBUTED_ARGS /mnt/huangyonghua/huangyonghua/mizar-checkpoint-engine/dlckpt/python/elastic_agent/torch/backup.py 2>&1 | tee ${LOG_BACKUP_NAME} &


echo "跑完"