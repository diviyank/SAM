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
data = pd.read_csv("example_data.csv")
output = sam.predict(data)
```
