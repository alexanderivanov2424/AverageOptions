from continuous_exp.ballOptions import *
from sklearn_extra.cluster import KMedoids


"""
S - list of states
D - pair wise distances between states as matrix. 
k - number of options
"""
def ContApproxAverageOptions(states, k):
    kmedoids = KMedoids(n_clusters=k+1, random_state=0, method="alternate").fit(states)

    centers = kmedoids.cluster_centers_

    options = []
    rootData = (centers[0][:15], centers[0][15:])
    for i in range(1, len(centers)):
        endData = (centers[i][:15], centers[i][15:])
        R = 1
        op = Option(rootData, endData, R)
        options.append(op)
        op = Option(endData, rootData, R)
        options.append(op)

    return options



def test():
    pass