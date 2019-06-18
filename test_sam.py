#!/usr/bin/env python

import cdt
from sam.sam import SAM

d, g = cdt.data.load_dataset('sachs')
m = SAM()
m.predict(d, nruns=1)
