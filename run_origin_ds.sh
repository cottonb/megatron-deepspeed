# export https_proxy=http://10.10.20.100:1089 http_proxy=http://10.10.20.100:1089 all_proxy=socks5://10.10.20.100:1089
#!/bin/bash
pip show sentencepiece || pip install sentencepiece
pip show datasets || pip install datasets
pip show py-cpuinfo || pip install py-cpuinfo
pip show hjson || pip install hjson
pip show transformers || pip install transformers

export PYTHONPATH=$PYTHONPATH:/mnt/huangyonghua/bupt/origin_ds
export ENABLE_ALL_OFFLOAD=1

export OMP_NUM_THREADS=10

export LOCAL_WORLD_SIZE=$(python3 -c 'import torch;print(torch.cuda.device_count())')


#!/bin/bash
DIR=`pwd`
###############################################################################
### Main configs
## GPT-3 models use 2K sequence length/context window
SEQ_LEN=2048

### The "GPT-3 XXX" below are configs from GPT-3 paper
### https://arxiv.org/abs/2005.14165, choose based on
### your desired model size or build your own configs

# 极小测试用
# MODEL_SIZE=0.125
# NUM_LAYERS=3
# HIDDEN_SIZE=24
# NUM_ATTN_HEADS=3
# GLOBAL_BATCH_SIZE=256
# LR=6.0e-4
# MIN_LR=6.0e-5

## GPT-3 Small 125M
# MODEL_SIZE=0.125
# NUM_LAYERS=12
# HIDDEN_SIZE=768
# NUM_ATTN_HEADS=12
# GLOBAL_BATCH_SIZE=256
# LR=6.0e-4
# MIN_LR=6.0e-5

## GPT-3 Small 125M
# MODEL_SIZE=0.125
# NUM_LAYERS=12
# HIDDEN_SIZE=768
# NUM_ATTN_HEADS=12
# GLOBAL_BATCH_SIZE=256
# LR=6.0e-4
# MIN_LR=6.0e-5

## GPT-3 Medium 350M
# MODEL_SIZE=0.35

## GPT-3 Large 760M
# MODEL_SIZE=0.76
# NUM_LAYERS=24
# HIDDEN_SIZE=1536
# NUM_ATTN_HEADS=16
# # GLOBAL_BATCH_SIZE=256
# GLOBAL_BATCH_SIZE=64
# LR=2.5e-4
# MIN_LR=2.5e-5



# ## GPT-3 XL 1.3B
# MODEL_SIZE=1.3
# NUM_LAYERS=24
# HIDDEN_SIZE=2048
# NUM_ATTN_HEADS=16
# GLOBAL_BATCH_SIZE=512
# LR=2.0e-4
# MIN_LR=2.0e-5

## GPT-3 XL 1.3B x 2
# MODEL_SIZE=1.3
# NUM_LAYERS=48
# HIDDEN_SIZE=2048
# NUM_ATTN_HEADS=16
# GLOBAL_BATCH_SIZE=512
# LR=2.0e-4
# MIN_LR=2.0e-5

## GPT-3 2.7B 缩小了global batch 
# MODEL_SIZE=2.7
# NUM_LAYERS=32
# HIDDEN_SIZE=2560
# NUM_ATTN_HEADS=32
# GLOBAL_BATCH_SIZE=64
# LR=1.6e-4
# MIN_LR=1.6e-5

## GPT-3 2.7B
# MODEL_SIZE=2.7
# NUM_LAYERS=32
# HIDDEN_SIZE=2560
# NUM_ATTN_HEADS=32
# GLOBAL_BATCH_SIZE=512
# LR=1.6e-4
# MIN_LR=1.6e-5

# ## GPT-3 6.7B
# MODEL_SIZE=6.7
# NUM_LAYERS=32
# HIDDEN_SIZE=4096
# NUM_ATTN_HEADS=32
# GLOBAL_BATCH_SIZE=128
# LR=1.2e-4
# MIN_LR=1.2e-5

## GPT-3 6.7B 但层数翻倍，大小应该也翻倍了
# MODEL_SIZE=13
# NUM_LAYERS=64
# HIDDEN_SIZE=4096
# NUM_ATTN_HEADS=32
# GLOBAL_BATCH_SIZE=128
# LR=1.2e-4
# MIN_LR=1.2e-5

## GPT-3 6.7B
# MODEL_SIZE=6.7
# NUM_LAYERS=32
# HIDDEN_SIZE=4096
# NUM_ATTN_HEADS=32
# GLOBAL_BATCH_SIZE=1024
# LR=1.2e-4
# MIN_LR=1.2e-5

## GPT-3 13B
MODEL_SIZE=13
NUM_LAYERS=40
HIDDEN_SIZE=5120
NUM_ATTN_HEADS=40
GLOBAL_BATCH_SIZE=128
LR=1.0e-4
MIN_LR=1.0e-5

## GPT-3 13B
# MODEL_SIZE=13
# NUM_LAYERS=40
# HIDDEN_SIZE=5120
# NUM_ATTN_HEADS=40
# GLOBAL_BATCH_SIZE=1024
# LR=1.0e-4
# MIN_LR=1.0e-5

## GPT-3 26B
# MODEL_SIZE=13
# NUM_LAYERS=96
# HIDDEN_SIZE=5120
# NUM_ATTN_HEADS=40
# GLOBAL_BATCH_SIZE=128
# LR=1.0e-4
# MIN_LR=1.0e-5

## GPT-3 175B
# MODEL_SIZE=175
# NUM_LAYERS=96
# HIDDEN_SIZE=12288
# NUM_ATTN_HEADS=96
# GLOBAL_BATCH_SIZE=1536
# LR=0.6e-4
# MIN_LR=0.6e-5
###############################################################################
### Training duration configs
## The main termination condition, original GPT-3 paper trains for 300B tokens
## For MoE model, we found sometimes training a bit more to 330B tokens helps
TRAIN_TOKENS=30000000
# TRAIN_TOKENS=330000000000

## TRAIN_SAMPLES is another termination condition and also affect the number of 
## data samples to be indexed. Since we want to reach the TRAIN_TOKENS
## above, and techniques like curriculum learning has less token in some steps,
## so we just set this config large enough to make sure we have enough
## processed data and don't terminate by TRAIN_SAMPLES.
TRAIN_SAMPLES=$(( ${TRAIN_TOKENS} * 3 / ${SEQ_LEN} ))

## Another termination condition in minutes. Set it large enough to avoid
## undesired early termination.
EXIT_DURATION=30000000
###############################################################################
### LR configs
## LR warmup and decay duration, this token-based config is preferable since
## no need to readjust when the batch size/seqlen is changed.
## Original GPT-3 paper uses 375M warmup tokens and 260B decay tokens.
## For MoE model, we found that setting the decay token to 300B helps.
WARMUP_TOKENS=37500
LR_DECAY_TOKENS=26000000
# LR_DECAY_TOKENS=300000000000

# 并行配置
###############################################################################
### Parallelism configs
## Micro batch size per GPU
## Make sure that BATCH_SIZE <= GLOBAL_BATCH_SIZE*PP_SIZE*MP_SIZE/NUM_GPUS
BATCH_SIZE=4
# BATCH_SIZE=1

## Model parallelism, 1 is no MP
MP_SIZE=1

## Pipeline parallelism
## Currently we don't support PP for MoE. To disable PP, set PP_SIZE
## to 1 and use the "--no-pipeline-parallel" arg.
PP_SIZE=1
NUM_GPUS=$LOCAL_WORLD_SIZE


total_gpus=$(($NUM_GPUS * $WORLD_SIZE))
mid_batch_size=$(($total_gpus * $BATCH_SIZE))
# mid_batch_size=$(($mid_batch_size * 4))
echo 节点数:$WORLD_SIZE
echo 总gpu数:$total_gpus
echo 中等batch_size:$mid_batch_size
###############################################################################
### MoE configs
## Number of experts. EP_SIZE 1 means dense model without MoE
EP_SIZE=1
# EP_SIZE=128

if [[ $EP_SIZE -gt $NUM_GPUS ]]; then
    EP_PARALLEL_SIZE=$NUM_GPUS
else
    EP_PARALLEL_SIZE=$EP_SIZE
fi

## Original GPT-3 model always set min LR at 10% of max LR. For MoE model, we
## found that lower LR and min LR (than the base dense model) helps.
## For 1.3B MoE-128 model we used LR=1.2e-4 and MIN_LR=1.0e-6.
## For 350M MoE-128 model we used LR=2.0e-4 and MIN_LR=2.0e-6, but they are not
## heavily tuned.
# LR=2.0e-4
# MIN_LR=2e-06

## Coefficient for MoE loss. We find that 0.01 is a good value at least for
## 1.3B MoE-128 model
MLC=0.01

## Below configs adjust the MoE expert token capacity limit during training and
## eval. To completely disable capacity limit, set MOE_DROP_TOKEN to false.
## Larger capacity factor or disabling capacity limit could improve training
## convergence, but will also reduce training throughput.
MOE_TRAIN_CAP_FACTOR=1.0
MOE_EVAL_CAP_FACTOR=1.0
MOE_MIN_CAP=4
MOE_DROP_TOKEN="true"
# MOE_DROP_TOKEN="false"
###############################################################################
### Curriculum learning (CL) configs
## Enable/disable CL
CL_ENABLED="false"
## Consult the tutorial https://www.deepspeed.ai/tutorials/curriculum-learning/
## for tuning the following configs
CL_START_SEQLEN=80
CL_AVG_SEQLEN=$(( (${CL_START_SEQLEN} + ${SEQ_LEN}) / 2 ))
CL_TOKENS=60
CL_TOKENS=$((${CL_TOKENS} * 1000000000))
CL_STEP=$(( ${CL_TOKENS} / (${GLOBAL_BATCH_SIZE} * ${CL_AVG_SEQLEN}) ))
###############################################################################
### Misc configs
LOG_INTERVAL=1
EVAL_ITERS=10
EVAL_INTERVAL=100
SAVE_INTERVAL=1000

## Standard deviation for weight initialization
## We used 0.014 for 350M/1.3B dense/MoE models, and used 0.01 for 6.7B
## dense model. Usually larger model needs lower std.
INIT_STD=0.014
# INIT_STD=0.01

## Activation checkpointing saves GPU memory, but reduces training speed
# ACTIVATION_CHECKPOINT="true"
ACTIVATION_CHECKPOINT="false"
###############################################################################
### Output and data configs
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
host="${HOSTNAME}"
NAME="gpt-${MODEL_SIZE}B-lr-${LR}-minlr-${MIN_LR}-bs-${GLOBAL_BATCH_SIZE}-gpus-${NUM_GPUS}-mp-${MP_SIZE}-pp-${PP_SIZE}"
if [[ $EP_SIZE -gt 1 ]]; then
    NAME="${NAME}-ep-${EP_SIZE}-mlc-${MLC}-cap-${MOE_TRAIN_CAP_FACTOR}-drop-${MOE_DROP_TOKEN}"
fi
if [ "${CL_ENABLED}" = "true" ]; then
    NAME="${NAME}-cl-${CL_START_SEQLEN}-${CL_STEP}"
fi

OUTPUT_BASEPATH=$DIR/output
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/log/"
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${NAME}_${host}_${current_time}"
mkdir -p ${TENSORBOARD_DIR} 
## Note that for MoE model with billion-scale base model, the checkpoint can be
## as large as TB-scale which normal NFS cannot handle efficiently.
CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${NAME}"


VOCAB_PATH=/mnt/huangyonghua/bupt/megatron-data/vocab.json
MERGE_PATH=/mnt/huangyonghua/bupt/megatron-data/merges.txt
# Public the Pile dataset, can be downloaded at https://mystic.the-eye.eu/public/AI/pile_neox/
DATA_BLEND=/mnt/huangyonghua/bupt/megatron-data/gpt2_text_document

###############################################################################
data_options=" \
         --vocab-file ${VOCAB_PATH} \
         --merge-file ${MERGE_PATH} \
         --data-path ${DATA_BLEND} \
         --data-impl mmap"

megatron_options=" \
        --override-opt_param-scheduler \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --tensor-model-parallel-size ${MP_SIZE} \
        --moe-expert-parallel-size ${EP_PARALLEL_SIZE} \
        --num-experts ${EP_SIZE} \
        --moe-loss-coeff ${MLC} \
        --moe-train-capacity-factor ${MOE_TRAIN_CAP_FACTOR} \
        --moe-eval-capacity-factor ${MOE_EVAL_CAP_FACTOR} \
        --moe-min-capacity ${MOE_MIN_CAP} \
        --init-method-std ${INIT_STD} \
        --lr-decay-tokens ${LR_DECAY_TOKENS} \
        --lr-warmup-tokens ${WARMUP_TOKENS} \
        --micro-batch-size ${BATCH_SIZE} \
        --exit-duration-in-mins ${EXIT_DURATION} \
        --rampup-batch-size ${mid_batch_size} ${mid_batch_size} 1953125 \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${SEQ_LEN} \
        --train-tokens ${TRAIN_TOKENS} \
        --train-samples ${TRAIN_SAMPLES} \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
        --split 98,2,0 \
        --log-interval ${LOG_INTERVAL} \
        --eval-interval ${EVAL_INTERVAL} \
        --eval-iters ${EVAL_ITERS} \
        --save-interval ${SAVE_INTERVAL} \
        --weight-decay 0.1 \
        --clip-grad 1.0 \
        --hysteresis 2 \
        --num-workers 0 \
        --fp16 \

        --recompute-granularity full \
        --recompute-method uniform \
        --recompute-num-layers 1 \

        --save ${CHECKPOINT_PATH} \
        --tensorboard-queue-size 1 \
        --log-timers-to-tensorboard \
        --timing-log-level 1 \
        --no-pipeline-parallel \
        
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --tensorboard-dir ${TENSORBOARD_DIR}"

# --cpu-optimizer \
# --load ${CHECKPOINT_PATH} \
# --rampup-batch-size 32 32 1953125 \

# 重计算用
# --recompute-activations 

# 完全重计算
# --recompute-granularity full \
# --recompute-method uniform \
# --recompute-num-layers 1 \

if [ "${ACTIVATION_CHECKPOINT}" = "true" ]; then
megatron_options="${megatron_options} \
        --checkpoint-activations"
fi

if [[ $EP_SIZE -gt 1 ]]; then
megatron_options="${megatron_options} \
        --create-moe-param-group"
fi

if [ "${MOE_DROP_TOKEN}" = "false" ]; then
megatron_options="${megatron_options} \
        --disable-moe-token-dropping"
fi

# template_json="ds_config_gpt_TEMPLATE.json"
template_json="origin_ds_config_template.json"
config_json="origin_ds_config_gpt_${NAME}.json"
echo 配置名:$config_json
sed "s/CONFIG_BATCH_SIZE/${GLOBAL_BATCH_SIZE}/" ${template_json} \
    | sed "s/CONFIG_MBSIZE/${BATCH_SIZE}/" \
    | sed "s/LOG_INTERVAL/${LOG_INTERVAL}/" \
    | sed "s/ZERO_STAGE/3/" \
    | sed "s/PRESCALE_GRAD/true/" \
    | sed "s/CONFIG_FP16_ENABLED/false/" \
    | sed "s/CONFIG_BF16_ENABLED/true/" \
    | sed "s/CONFIG_CL_ENABLED/${CL_ENABLED}/" \
    | sed "s/CONFIG_CL_MIN/${CL_START_SEQLEN}/" \
    | sed "s/CONFIG_CL_MAX/${SEQ_LEN}/" \
    | sed "s/CONFIG_CL_DURATION/${CL_STEP}/" \
	  > ${config_json}

deepspeed_options=" \
		    --deepspeed \
		    --deepspeed_config ${config_json} \
		    --pipeline-model-parallel-size ${PP_SIZE}"

# Currently MoE is not compatible with pipeline parallel
if [[ $EP_SIZE -gt 1 ]]; then
deepspeed_options="${deepspeed_options} \
        --no-pipeline-parallel"
fi

if [ "${ACTIVATION_CHECKPOINT}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
        --deepspeed-activation-checkpointing"
fi


NNODES=${WORLD_SIZE:-1}
export NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=9200
hostfile=./myhostfile
DISTRIBUTED_ARGS="
    --no_ssh \
    --num_nodes=$NNODES\
    --node_rank=$NODE_RANK\
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT\
    --num_gpus=$LOCAL_WORLD_SIZE\
    --hostfile=$hostfile"

# --bind_cores_to_rank
# --node_rank=$RANK\
# --num_gpus=$LOCAL_WORLD_SIZE\


ds=/mnt/huangyonghua/bupt/origin_ds/bin/deepspeed
run_name=/mnt/huangyonghua/bupt/Megatron-DeepSpeed/pretrain_gpt.py
run_cmd="python $ds ${DISTRIBUTED_ARGS} $run_name ${megatron_options} ${data_options} ${deepspeed_options}  |  tee ${OUTPUT_BASEPATH}/log/origin_${NAME}_${host}_${current_time}.log"
echo ${run_cmd}
eval ${run_cmd}
# set +x