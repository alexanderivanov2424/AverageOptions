
# options
from options.FiedlerOptions import FiedlerOptions
from options.EigenOptions import Eigenoptions
from options.AverageOptions import AverageShortestOptions
from options.ApproxAverageOptions import ApproxAverageOptions
from options.HittingOptions import HittingTimeOptions

def GetOptions(A, k, method='eigen'):
    print("Getting options by method: " + method)

    if method == 'eigen':
        B, options, vectors = Eigenoptions(A, k*2)
    elif method == 'fiedler':
        B, options, _, vectors = FiedlerOptions(A, k*2)
        options = [(opp[0][0],opp[1][0]) for opp in options]
    elif method == 'ASPDM':
        B, options = AverageShortestOptions(A, None, k)
        vectors = None
    elif method == 'ApproxAverage':
        B, options = ApproxAverageOptions(A, k)
        vectors = None
    elif method == 'hitting':
        B, options = HittingTimeOptions(A, k)
        vectors = None
    elif method == 'bet':
        # TODO: B is empty.
        B, options, vectors = BetweennessOptions(A, k)

    options.extend([(opp[1], opp[0]) for opp in options])
    return B, options, vectors
