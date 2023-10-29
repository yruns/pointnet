import numpy as np

def coordinates_normalize(coordinates):
    centroid = np.mean(coordinates, axis=0)
    coordinates = coordinates - centroid
    m = np.max(np.sqrt(np.sum(coordinates ** 2, axis=1)))
    coordinates /= m
    return coordinates
