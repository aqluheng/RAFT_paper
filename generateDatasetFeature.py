# +
import sys
import torch

sys.path.append("core")
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser("Generate Dataset Feature.")
parser.add_argument(
    "--dataset",
    choices=["sintel", "chairs", "movi", "MoviFilter", "things"],
    help="Dataset to analyse.",
    required=True,
)
parser.add_argument(
    "--start", "-s", type=int, default=0, help="Start index (default:0)."
)
parser.add_argument(
    "--len", type=int, default=1000, help="Process max length. (default:1000)"
)
parser.add_argument("--merge", action="store_true", help="Merge sub result.")
args = parser.parse_args()

import datasets

if args.dataset == "chairs":
    analyseDataset = datasets.FlyingChairs(split="training")
elif args.dataset == "sintel":
    analyseDataset = datasets.MpiSintel(
        None, split="training", dstype="clean"
    ) + datasets.MpiSintel(None, split="training", dstype="final")
elif args.dataset == "movi":
    analyseDataset = datasets.moviSubset()
elif args.dataset == "MoviFilter":
    analyseDataset = datasets.moviFilter()
elif args.dataset == "things":
    analyseDataset = datasets.FlyingThings3D(
        dstype="frames_cleanpass"
    ) + datasets.FlyingThings3D(dstype="frames_finalpass")

startIndex = args.start
endIndex = min(len(analyseDataset), args.start + args.len)
if startIndex >= len(analyseDataset):
    print(f"StartIndex:{startIndex} > DatasetSize:{len(analyseDataset)}")
    exit(0)

def mergeDataset(subDatasetList):
    velocityHist = torch.zeros((3000))
    directHist = torch.zeros((720))
    velocityStdList = []
    directStdList = []
    for subDataset in subDatasetList:
        velocityHist += subDataset["velocityHist"]
        directHist += subDataset["directionHist"]
        velocityStdList += list(subDataset["velocityStd"])
        directStdList += list(subDataset["directionStd"])

    velocityStdTensor = torch.Tensor(velocityStdList)
    directStdTensor = torch.Tensor(directStdList)
    result = {
        "velocityHist": velocityHist,
        "directionHist": directHist,
        "velocityStd": velocityStdTensor,
        "directionStd": directStdTensor,
    }
    resultFile = f"dataLog/{args.dataset}Dataset_final.pth"
    print(resultFile)
    torch.save(result, resultFile)


if args.merge:
    print("Merge")
    subDataset = []
    for startIndex in range(0, len(analyseDataset), 1000):
        loadFileName = f"dataLog/{args.dataset}Dataset_{startIndex}.pth"
        subDataset.append(torch.load(loadFileName))
    mergeDataset(subDataset)
else:
    resultFile = f"dataLog/{args.dataset}Dataset_{startIndex}.pth"
    print(f"Process {startIndex}-{endIndex}")

    velocityHist = torch.zeros((3000))
    directHist = torch.zeros((720))
    velocityStdList = []
    directStdList = []

    for idx in tqdm(range(startIndex, endIndex)):
        images, flow, valid = analyseDataset[idx]
        velocity = torch.sqrt(flow[0] ** 2 + flow[1] ** 2)
        direct = torch.arctan(flow[1] / (flow[0] + 1e-7))
        direct[flow[0] < 0] += torch.pi * 3 / 2
        direct[flow[0] >= 0] += torch.pi / 2
        directHist += torch.histogram(direct, bins=720, range=(0, torch.pi * 2))[0]
        velocityHist += torch.histogram(velocity, bins=3000, range=(0, 300))[0]
        velocityStdList.append(velocity.std().item())
        directStdList.append(direct.std().item())

    velocityStdTensor = torch.Tensor(velocityStdList)
    directStdTensor = torch.Tensor(directStdList)
    datasetDict = {
        "velocityHist": velocityHist,
        "directionHist": directHist,
        "velocityStd": velocityStdTensor,
        "directionStd": directStdTensor,
    }
    torch.save(datasetDict, resultFile)
