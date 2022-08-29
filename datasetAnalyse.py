# +
import sys
import torch
sys.path.append("core")
import datasets
analyseDataset = datasets.moviSubset()

velocityHist = torch.zeros((3000))
directHist = torch.zeros((720))
velocityStdList = []
directStdList = []
for idx, dataTuple in enumerate(analyseDataset):
    images, flow, valid = dataTuple
    velocity = torch.sqrt(flow[0]**2+flow[1]**2)
    direct = torch.arctan(flow[1]/(flow[0]+1e-7))
    direct[flow[0]<0] += (torch.pi*3/2)
    direct[flow[0]>=0] += (torch.pi/2)
    directHist += torch.histogram(direct,bins=720,range=(0,torch.pi*2))[0]
    velocityHist += torch.histogram(velocity,bins=3000,range=(0,300))[0]
    velocityStdList.append(velocity.std().item())
    directStdList.append(direct.std().item())

velocityStdTensor = torch.Tensor(velocityStdList)
directStdTensor = torch.Tensor(directStdList)
datasetDict = {"velocityHist": velocityHist, "directionHist": directHist, "velocityStd": velocityStdTensor, "directionStd": directStdTensor}
torch.save(datasetDict,"dataLog/movi.pth")
