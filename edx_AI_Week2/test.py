#!/usr/bin/python -tt
# Copyright 2010 Google Inc.
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0

# Google's Python Class
# http://code.google.com/edu/languages/google-python-class/

import driver
import sys
import Queue as queue
import time

class aclass:

    def __init__(self, nsize):
        self.nsize = nsize
        self.tsize = pow(self.nsize, 2)
        self.list = []
        self.list.append([1, -1, self.nsize, -self.nsize])
        self.list.append([1, -self.nsize, -1, self.nsize])
        self.list.append([1, -self.nsize, self.nsize, -1])
        self.list.append([1, -1, -self.nsize, self.nsize])
        self.list.append([1, self.nsize, -self.nsize, -1])
        self.list.append([1, self.nsize, -1, -self.nsize])
                                    
        self.list.append([-self.nsize, -1, self.nsize, 1])
        self.list.append([-self.nsize, -1, 1, self.nsize])
        self.list.append([-self.nsize, 1, -1, self.nsize])
        self.list.append([-self.nsize, 1, self.nsize, -1])
        self.list.append([-self.nsize, self.nsize, 1, -1])
        self.list.append([-self.nsize, self.nsize, -1, 1])
                                        
        self.list.append([-1, 1, -self.nsize, self.nsize])
        self.list.append([-1, 1, self.nsize, -self.nsize])
        self.list.append([-1, self.nsize, 1, -self.nsize])
        self.list.append([-1, self.nsize, -self.nsize, 1])
        self.list.append([-1, -self.nsize, self.nsize, 1])
        self.list.append([-1, -self.nsize, 1, self.nsize])
                                        
        self.list.append([self.nsize, -1, -self.nsize, 1])
        self.list.append([self.nsize, -1, 1, -self.nsize])
        self.list.append([self.nsize, 1, -1, -self.nsize])
        self.list.append([self.nsize, 1, -self.nsize, -1])
        self.list.append([self.nsize, -self.nsize, 1, -1])
        self.list.append([self.nsize, -self.nsize, -1, 1])
        return

    def printIt(self):
        for i in self.list:
            print i
        return

    def callIt(self):
        start = [1,2,5,3,4,0,6,7,8]
        for moves in self.list:
            state = driver.State(3, "dfs", moves)
            print "\n" + "*" * 40
            print "ast results for " + str(moves)
            state.solve_it(start)
            print "*" * 40
            #if (state.nodes_expanded == 181437 and 
            #    state.max_fringe_size == 42913 and 
            #    state.max_search_depth == 66125):
            #    return
        
def main():
    my_states = aclass(3)
    my_states.callIt()
    

    #prio_queue = queue.PriorityQueue()
    #prio_queue.put((2, time.time(), 'super blah'))
    #time.sleep(0.1)
    #prio_queue.put((1, time.time(), 'This thing would come after Some Thing if we sorted by this text entry'))
    #time.sleep(0.1)
    #prio_queue.put((1, time.time(), 'Some thing'))
    #time.sleep(0.1)
    #prio_queue.put((5, time.time(), 'blah'))
    #
    #while not prio_queue.empty():
    #    #item = prio_queue.get()
    #    (number, hour, text) = prio_queue.get()
    #    #print('%s.%s - %s' % item)
    
       
if __name__ == '__main__':
    main()
