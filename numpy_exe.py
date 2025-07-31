import numpy as np
import cv2 as cv

class Array:
    @staticmethod
    def numpy_2d_array():
        a = np.array([1, 2, 3, 4, 5, 6])
        f = np.array([7, 8, 9])
        return np.concatenate((a, f))



arr = Array()

def main():
    print(arr.numpy_2d_array())

main()
