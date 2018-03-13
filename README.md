# SAM: Structural Agnostic Model, Causal Discovery and Penalized Adversarial Learning.
Code in PyTorch.  

In order to use SAM:
1. Install the package requirements with ```pip install -r requirements.txt```. For PyTorch visit: http://pytorch.org
2. Install the package with the command: ```python setup.py develop --user ```
3. Execute the SAM by including the desired options:
```python
import pandas as pd
from sam import SAM
sam = SAM()
data = pd.read_csv("datasets/graph_train/G5_v1_numdata.tab", sep="\t")
output = sam.predict(data, nruns=12) # Recommended if high computational capability available, else nruns=1
```

In order to download the datasets used in the paper as well as the generators, download the submodule "datasets" (458MB):
```
git submodule update --init
```

The acyclic graphs for the mechanisms _Linear, GP Add, GP Mix, Sigmoid Add and Sigmoid Mix_ were generated using the software provided at : https://github.com/bquast/ANM/tree/master/codeANM
