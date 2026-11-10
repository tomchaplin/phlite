from phlite_grpph import grpph
import logging
from pprint import pprint

FORMAT = "%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s"
logging.basicConfig(format=FORMAT)
logging.getLogger().setLevel(logging.INFO)

barcode = grpph(
    6,
    [
        (0, 1, 10.0),
        (0, 2, 10.0),
        (1, 3, 10.0),
        (2, 3, 10.0),
        (0, 3, 20.0),
        (0, 4, 5.0),
        (4, 5, 10.0),
        (5, 3, 5.0),
    ],
    representatives=True,
)

print(barcode)
print(barcode.essential)
print(barcode.pairings)
# Representatives line up with the bars: pairing_representatives[1][i] represents pairings[1][i]
for bar, rep in zip(barcode.pairings[1], barcode.pairing_representatives[1]):
    print(bar, rep)
