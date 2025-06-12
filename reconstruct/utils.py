import numpy as np

def dihedral(b1, b2, b3):
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    u2 = b2 / np.linalg.norm(b2)
    x = np.dot(n1, n2) if len(n2.shape) == 1 else np.dot(n2, n1)
    y = np.dot(np.cross(n1, n2), u2)
    return np.arctan2(y, x)

def rad2deg(radians):
    return radians * (180.0 / np.pi)