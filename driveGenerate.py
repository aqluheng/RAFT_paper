# -*- coding: utf-8 -*-
# ### 生成DatasetFeature的代码

import os
for i in range(9000,131951,2000):
    print(i)
    os.system("python generateDatasetFeature.py --dataset movi -s %d &"%i)
    os.system("python generateDatasetFeature.py --dataset movi -s %d"%(i+1000))

import os
for i in range(0,22232,2000):
    print(i)
    os.system("python generateDatasetFeature.py --dataset chairs -s %d &"%i)
    os.system("python generateDatasetFeature.py --dataset chairs -s %d"%(i+1000))

import os
for i in range(0, 8132, 2000):
    print(i)
    os.system("python generateDatasetFeature.py --dataset MoviFilter -s %d &"%i)
    os.system("python generateDatasetFeature.py --dataset MoviFilter -s %d"%(i+1000))

import os
for i in range(0, 80604, 2000):
    print(i)
    os.system("python generateDatasetFeature.py --dataset things -s %d &"%i)
    os.system("python generateDatasetFeature.py --dataset things -s %d"%(i+1000))

# ### 生成PerFlow的代码

for i in range(0, 2082, 1000):
    os.system("python generatePerFlowFeature.py --dataset sintel -s %d &"%i)

import os
for i in range(7000,131951,2000):
    print(i)
    os.system("python generatePerFlowFeature.py --dataset movi -s %d &"%i)    
    os.system("python generatePerFlowFeature.py --dataset movi -s %d"%(i+1000))

import os
for i in range(0,22232,2000):
    print(i)
    os.system("python generatePerFlowFeature.py --dataset chairs -s %d"%(i+1000))
#     os.system("python generatePerFlowFeature.py --dataset chairs -s %d"%i)


