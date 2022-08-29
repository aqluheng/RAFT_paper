PROJECT="RAFT_diffModel3"
APIKEY="09c6e5e2dbd5e78ff164ef5477b02778d42ce95f"
MODELCFG=$1
DATASET=$2
AUGPARAM=$3

# +
# echo "DATASET:${DATASET} MODELCFG:${MODELCFG} AUGPARAM:${AUGPARAM}"
if [[ ${MODELCFG} = "big" ]]; then
    MODELSET=""
fi
if [[ ${MODELCFG} = "small" ]]; then
    MODELSET="--small"
fi

if [[ ${AUGPARAM} = "autoflowAug" ]]; then
    HYPERSET=""
fi
if [[ ${AUGPARAM} = "kittiAug" ]]; then
    HYPERSET=""
fi

NAME="raft-${MODELCFG}-${DATASET}-${AUGPARAM}-200k-lr0.4"
# -

RESULTFILE="evaluateLog/${NAME}.log"
pip install wandb scikit-image tensorflow
FILE=checkpoints/${NAME}.pth
if [ ! -f "$FILE" ]; then
    echo "python -u train.py --name ${NAME} --stage ${DATASET} ${MODELSET} --aug_setting ${AUGPARAM} --validation sintel --num_steps 200000 --batch_size 10 --lr 0.0004 --image_size 512 512 --wdecay 0.0001 --gpus 0 --enable_wandb" 
    echo "python -u train.py --name ${NAME} --stage ${DATASET} ${MODELSET} --aug_setting ${AUGPARAM} --validation sintel --num_steps 200000 --batch_size 10 --lr 0.0004 --image_size 512 512 --wdecay 0.0001 --gpus 0 --enable_wandb" >> ${RESULTFILE}
    WANDB_API_KEY=${APIKEY} WB_PROJECT=${PROJECT} WB_ID=0 python -u train.py --name ${NAME} --stage ${DATASET} ${MODELSET} --aug_setting ${AUGPARAM} --validation sintel --num_steps 200000 --batch_size 10 --lr 0.0004 --image_size 512 512 --wdecay 0.0001 --gpus 0 --enable_wandb
fi
if [ -f "$FILE" ]; then
    echo ${FILE} >> ${RESULTFILE}
    # OMP_NUM_THREADS=8 python evaluate.py ${MODELSET} --model ${FILE} --dataset kitti --iters 12 >> ${RESULTFILE}
    OMP_NUM_THREADS=8 python evaluate.py ${MODELSET} --model ${FILE} --dataset sintel --iters 12 >> ${RESULTFILE}
fi
