PROJECT="RAFT_diffModel3"
APIKEY="09c6e5e2dbd5e78ff164ef5477b02778d42ce95f"
MODELCFG=$1
DATASET=$2

echo "DATASET:${DATASET} MODELCFG:${MODELCFG}"
if [[ ${MODELCFG} = "big" ]]; then
    MODELSET=""
fi
if [[ ${MODELCFG} = "small" ]]; then
    MODELSET="--small"
fi

NAME="raft-${MODELCFG}-${DATASET}-${AUGPARAM}"
RESULTFILE="evaluateLog/raft-${MODELCFG}-${DATASET}-${AUGPARAM}.log"
pip install wandb scikit-image
FILE=checkpoints/${NAME}.pth
if [ ! -f "$FILE" ]; then
    WANDB_API_KEY=${APIKEY} WB_PROJECT=${PROJECT} WB_ID=0 python -u train.py --name ${NAME} --stage ${DATASET} ${MODELSET} --validation sintel --num_steps 100000 --batch_size 10 --lr 0.0004 --image_size 512 512 --wdecay 0.0001 --gpus 0
fi
if [ -f "$FILE" ]; then
    echo ${FILE} >> ${RESULTFILE}
    OMP_NUM_THREADS=8 python evaluate.py ${MODELSET} --model ${FILE} --dataset kitti --iters 6 >> ${RESULTFILE}
    OMP_NUM_THREADS=8 python evaluate.py ${MODELSET} --model ${FILE} --dataset sintel --iters 6 >> ${RESULTFILE}
fi