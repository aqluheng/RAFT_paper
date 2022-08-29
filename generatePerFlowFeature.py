# +
import torch


def getVelocityFeature(flow):
    velocityList = torch.sqrt(flow[0] ** 2 + flow[1] ** 2).reshape(-1)
    sortedVelocity = torch.sort(velocityList)[0]
    velocityLen = len(velocityList)
    quantilePoint = [
        sortedVelocity[velocityLen // 4],
        sortedVelocity[velocityLen // 2],
        sortedVelocity[3 * velocityLen // 4],
        sortedVelocity[-1],
    ]
    velocityMAD = (velocityList - velocityList.mean()).abs().mean()
    return quantilePoint, [torch.std(velocityList)] + [velocityMAD]


def getDirectFeature(flow):
    directList = torch.arctan(flow[1] / (flow[0] + 1e-6))
    directList[flow[0] < 0] += torch.pi
    degreeDistribute = torch.histogram(
        directList, bins=8, range=(-torch.pi / 2, torch.pi / 2 * 3)
    )[0]
    degreeDistribute /= degreeDistribute.sum()
    directMAD = (directList - directList.mean()).abs().mean()
    return degreeDistribute, [torch.std(directList)] + [directMAD]


# +
import sys
import torch

sys.path.append("core")
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser("Generate PerFlow Feature.")
parser.add_argument(
    "--dataset",
    choices=["sintel", "chairs", "movi"],
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

startIndex = args.start
endIndex = min(len(analyseDataset), args.start + args.len)
if startIndex >= len(analyseDataset):
    print(f"StartIndex:{startIndex} > DatasetSize:{len(analyseDataset)}")
    exit(0)


def mergePerFlow(subPerFlowList):
    result = []
    for subPerFlow in subPerFlowList:
        result += subPerFlow
    resultFile = f"dataLog/{args.dataset}PerFlow_final.pth"
    print(resultFile)
    torch.save(result, resultFile)


if args.merge:
    print("Merge")
    subPerFlow = []
    for startIndex in range(0, len(analyseDataset), 1000):
        loadFileName = f"dataLog/{args.dataset}PerFlow_{startIndex}.pth"
        subPerFlow.append(torch.load(loadFileName))
    mergePerFlow(subPerFlow)


else:
    resultFile = f"dataLog/{args.dataset}PerFlow_{startIndex}.pth"
    print(f"Process {startIndex}-{endIndex}")
    allFeature = []
    for idx in tqdm(range(startIndex, endIndex)):
        images, flow, valid = analyseDataset[idx]
        velocityQuantile, velocityStd = getVelocityFeature(flow)
        directDistribute, directStd = getDirectFeature(flow)
        allFeature.append(
            [
                torch.Tensor(velocityQuantile),
                torch.Tensor(velocityStd),
                torch.Tensor(directDistribute),
                torch.Tensor(directStd),
            ]
        )
    torch.save(allFeature, resultFile)
