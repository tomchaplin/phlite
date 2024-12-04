# This tests that we can support passing in edges with infinite weight
# as well as isolated vertices and multiple connected components

import numpy as np
from phlite_grpph import grpph_with_involution
import logging
from pprint import pprint

FORMAT = "%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s"
logging.basicConfig(format=FORMAT)
logging.getLogger().setLevel(logging.INFO)

essential, pairings, reps = grpph_with_involution(
    6,
    [
        (0, 1, 1.0),
        (1, 0, 1.0),
        (1, 2, np.inf),
        (3, 4, 10.0),
        (4, 3, 10.0),
    ],
)

print(essential)
print(pairings)
pprint(reps)
