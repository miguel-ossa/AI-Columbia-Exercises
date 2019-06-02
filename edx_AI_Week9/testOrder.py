#!/usr/bin/python

"""
  Implementation of the IA for the Sudoku problem,
  using AC-3 and Backtracking

  Miguel de la Ossa July, 2017
"""
import sys
import copy
from collections import deque, OrderedDict
from heapq import heapify, heappush, heappop
import time

_DEBUG_LEVEL = 1


def main(script, *args):
    
    # prueba velocidad
    # set
    initialTime = time.clock()
    
    D = set()
    for i in range(0,10000000):
        D.add((1, i))
    for i in D:
        x = i

    finalTime = time.clock()
    print ('time set', finalTime - initialTime) # para set = 6.3304
                                    
    # ist
    initialTime = time.clock()
    
    D = []
    for i in range(0,10000000):
        D.append([1, i])
    for i in D:
        x = i

    finalTime = time.clock()
    print ('time list', finalTime - initialTime) # para set = 25.273460999999998
                                    

if __name__ == '__main__':
    main(*sys.argv)

