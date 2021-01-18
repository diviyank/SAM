# Structural Agnostic Modeling: Adversarial Learning of Causal Graphs
This version is the new version of SAM, using structural gates and functional
gates
## NB: This code is the code of the V2 of SAM, for the lastest (V3), please check the CDT package (https://github.com/FenTechSolutions/CausalDiscoveryToolbox) in the time being
Code in PyTorch. Link to the paper: https://arxiv.org/abs/1803.04929

The (OLD) code of SAM is available at
https://github.com/Diviyan-Kalainathan/SAMv1
The first version of the paper is available at https://arxiv.org/abs/1803.04929v1

In order to use SAM:
1. Install the package requirements with ```pip install -r requirements.txt```. For PyTorch visit: http://pytorch.org
2. Install the package with the command: ```python setup.py develop --user ```
3. Execute the SAM by including the desired options:
```python
import pandas as pd
from sam import SAM
sam = SAM()
data = pd.read_csv("test/G5_v1_numdata.tab", sep="\t")
output = sam.predict(data, nruns=12) # Recommended if high computational capability available, else nruns=1
```

We highly recommand to use GPUs if possible. Here is an example for 2 GPUs:
```python
import pandas as pd
from sam import SAM
sam = SAM()
data = pd.read_csv("test/G5_v1_numdata.tab", sep="\t")
output = sam.predict(data, nruns=12, gpus=2, njobs=4) # As the model is small, we recommand using 2 jobs on each GPU
```


In order to download the datasets used in the paper as well as the generators, download the submodule "datasets" (458MB):
```
git submodule update --init
```

The acyclic graphs for the mechanisms _Linear, GP Add, GP Mix, Sigmoid Add and Sigmoid Mix_ were generated using the software provided at : https://github.com/bquast/ANM/tree/master/codeANM

