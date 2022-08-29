PROJECT="RAFT_diffModel3"
APIKEY="09c6e5e2dbd5e78ff164ef5477b02778d42ce95f"
MODELCFG=$1
DATASET=$2
AUGPARAM=$3
ITERS=$4

# +
echo "DATASET:${DATASET} MODELCFG:${MODELCFG} AUGPARAM:${AUGPARAM}"
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

NAME="raft-${MODELCFG}-${DATASET}-${AUGPARAM}"
# -

RESULTFILE="evaluateLog/${NAME}.log"
pip install wandb scikit-image tensorflow
FILE=checkpoints/${NAME}.pth
if [ -f "$FILE" ]; then
    echo "python -u evaluateDiffModel.py --name ${NAME} --iters ${ITERS} ${MODELSET} --validation sintel --enable_wandb" 
    echo "python -u evaluateDiffModel.py --name ${NAME} --iters ${ITERS} ${MODELSET} --validation sintel --enable_wandb" >> ${RESULTFILE}
    WANDB_API_KEY=${APIKEY} WB_PROJECT=${PROJECT} WB_ID=0 python -u evaluateDiffModel.py --name ${NAME} --iters ${ITERS} ${MODELSET} --validation sintel --enable_wandb
else
    echo "$FILE not found."
fi
