#!/usr/bin/python

import sys
import ast
from collections import deque
import time
import resource
from heapq import heapify, heappush, heappop

_DEBUG_LEVEL = 1

    
class priority_dict(dict):
    """Dictionary that can be used as a priority queue.

    Keys of the dictionary are items to be put into the queue, and values
    are their respective priorities. All dictionary methods work as expected.
    The advantage over a standard heapq-based priority queue is
    that priorities of items can be efficiently updated (amortized O(1))
    using code as 'thedict[item] = new_priority.'

    The 'smallest' method can be used to return the object with lowest
    priority, and 'pop_smallest' also removes it.

    The 'sorted_iter' method provides a destructive sorted iterator.
    """
    
    def __init__(self, *args, **kwargs):
        super(priority_dict, self).__init__(*args, **kwargs)
        self._rebuild_heap()

    def _rebuild_heap(self):
        self._heap = [(v, k) for k, v in self.iteritems()]
        heapify(self._heap)

    def smallest(self):
        """Return the item with the lowest priority.

        Raises IndexError if the object is empty.
        """
        
        heap = self._heap
        v, k = heap[0]
        while k not in self or self[k] != v:
            heappop(heap)
            v, k = heap[0]
        return k

    def pop_smallest(self):
        """Return the item with the lowest priority and remove it.

        Raises IndexError if the object is empty.
        """
        
        heap = self._heap
        v, k = heappop(heap)
        while k not in self or self[k] != v:
            v, k = heappop(heap)
        del self[k]
        return k

    def __setitem__(self, key, val):
        # We are not going to remove the previous value from the heap,
        # since this would have a cost O(n).
        
        super(priority_dict, self).__setitem__(key, val)
        
        if len(self._heap) < 2 * len(self):
            heappush(self._heap, (val, key))
        else:
            # When the heap grows larger than 2 * len(self), we rebuild it
            # from scratch to avoid wasting too much memory.
            self._rebuild_heap()

    def setdefault(self, key, val):
        if key not in self:
            self[key] = val
            return val
        return self[key]

    def update(self, *args, **kwargs):
        # Reimplementing dict.update is tricky -- see e.g.
        # http://mail.python.org/pipermail/python-ideas/2007-May/000744.html
        # We just rebuild the heap from scratch after passing to super.
        
        super(priority_dict, self).update(*args, **kwargs)
        self._rebuild_heap()

    def sorted_iter(self):
        """Sorted iterator of the priority dictionary items.

        Beware: this will destroy elements as they are returned.
        """
        
        while self:
            yield self.pop_smallest()
            
class node_object(object):
    __slots__ = ['parent', 'depth', 'move', 'heuristic']
    def __str__(self):
        return 'depth: %d move: %d heuristic: %d' % (self.depth, self.move, self.heuristic)
    
class State:
    def __init__(self, nsize, algorithm): #, movs=[]):
        """Initialize the n-puzzle problem, with n-size value, tsize the total nodes and initial the goal state from n.
        """

        self.nsize = nsize
        self.tsize = pow(self.nsize, 2)
        self.goal = range(0, self.tsize)
        self.algorithm = algorithm
        self.pexpands = {}
            
        if self.algorithm == "bfs":
            self.frontier = deque()
            self.nodes = deque()
            self.moves = [-self.nsize, self.nsize, -1, 1]
        elif self.algorithm == "dfs":
            self.frontier = deque()
            self.nodes = deque()
            self.moves = [1, -1, self.nsize, -self.nsize]
        else:
            self.frontier = priority_dict()
            self.nodes = dict()
            self.moves = [-self.nsize, self.nsize, -1, 1]

        for key in range(self.tsize):
            self.pexpands[key] = self.getvalues(key)
            
        self.explored = set()

        self.nodes_expanded = 0 #the number of nodes that have been expanded
        #self.max_fringe_size = 0 #the maximum size of the frontier set in the lifetime of the algorithm
        self.max_search_depth = 0 #the maximum depth of the search tree in the lifetime of the algorithm

    def printst(self, st):
        """Print the list in a Matrix Format."""

        for (index, value) in enumerate(st):
            print ' %s ' % value, 
            if index in [x for x in range(self.nsize - 1, self.tsize, 
                         self.nsize)]:
                print 
        print 

    """Para cada key (de 0 a nsize) se comprueban los posibles movimientos en sus 
   casillas de partida.

   /-----------\
   | 0 | 1 | 2 |
   |-----------|
   | 3 | 4 | 5 |
   |-----------|
   | 6 | 7 | 8 |
   \-----------/

   Por ejemplo, si sumamos 1 al 0, lo desplazamos a la casilla 1. Si le sumamos 3, 
   ira a parar a la casilla de abajo: la 3.  
    """
    def getvalues(self, key):
        valid = []
        for x in self.moves:
            if 0 <= key + x < self.tsize: # up, down
                # [2, 5, 8] right
                if x == 1 and key in range(self.nsize - 1, self.tsize, 
                        self.nsize):
                    continue
                # [0, 3, 6] left
                if x == -1 and key in range(0, self.tsize, self.nsize):
                    continue
                valid.append(x)
        return tuple(valid)

    def neighbors(self, st):

        pos = st.index(0)

        #check possible movs at this position
        moves = self.pexpands[pos]
        expstates = []
        for mv in moves:
            nstate = st[:]
            (nstate[pos + mv], nstate[pos]) = (nstate[pos], nstate[pos + mv])
            expstates.append(nstate)
        return zip(expstates, moves)

    def manhattan_distance(self, st, target):
        mdist = 0
        for node in st:
            if node != 0:
                gdist = abs(target.index(node) - st.index(node))
                (jumps, steps) = (gdist // self.nsize, gdist % self.nsize)
                mdist += jumps + steps
        return mdist

    def new_node(self, depth, parent):
        x = node_object()
        x.depth = depth
        x.parent = parent
        x.move = 0
        return x
    
    def child_node(self, parent, move):
        x = node_object()
        x.depth = parent.depth + 1
        x.parent = parent
        x.move = move
        return x
    
    def uninformed_search_bfs(self, initial_state):
        self.frontier.append(str(initial_state))
        node = self.new_node(0, None)
        self.nodes.append(node)
        #self.max_fringe_size = 1
        while self.frontier:
            state = ast.literal_eval(self.frontier.popleft())
            self.explored.add(tuple(state))
            node = self.nodes.popleft()
            if state == self.goal:
                return node
            self.nodes_expanded += 1
            for neighbor, move in self.neighbors(state):
                if not tuple(neighbor) in self.explored and not str(neighbor) in self.frontier:
                    self.frontier.append(str(neighbor))
                    self.nodes.append(self.child_node(node, move))
                    if node.depth + 1 > self.max_search_depth:
                        self.max_search_depth = node.depth + 1
        return

    def uninformed_search_dfs(self, initial_state):
        self.frontier.append(str(initial_state))
        node = self.new_node(0, None)
        self.nodes.append(node)
        #self.max_fringe_size = 1
        while self.frontier:
            state = ast.literal_eval(self.frontier.pop())
            self.explored.add(tuple(state))
            node = self.nodes.pop()
            if state == self.goal:
                return node
            self.nodes_expanded += 1
            for neighbor, move in self.neighbors(state):
                if not tuple(neighbor) in self.explored and not str(neighbor) in self.frontier:
                    self.frontier.append(str(neighbor))
                    self.nodes.append(self.child_node(node, move))
                    if node.depth + 1 > self.max_search_depth:
                        self.max_search_depth = node.depth + 1
        return

    def ast_search(self, initial_state):
        self.frontier[tuple(initial_state)] = 0
        node = self.new_node(0, None)
        node.heuristic = 0
        self.nodes[tuple(initial_state)] = node
        #self.max_fringe_size = 1
        while self.frontier:
            state = list(self.frontier.pop_smallest())
            self.explored.add(tuple(state))
            node = self.nodes[tuple(state)]
            if state == self.goal:
                return node
            self.nodes_expanded += 1
            for neighbor, move in self.neighbors(state):
                if not tuple(neighbor) in self.explored:
                    child = self.child_node(node, move)
                    child.heuristic = node.heuristic + self.manhattan_distance(state, neighbor)
                    dist = child.heuristic + self.manhattan_distance(neighbor, self.goal) 
                    if not tuple(neighbor) in self.frontier:
                        self.frontier[tuple(neighbor)] = dist 
                        self.nodes[tuple(neighbor)] = child
                        if child.depth > self.max_search_depth:
                            self.max_search_depth = child.depth
                    else:
                        if self.frontier[tuple(neighbor)] > dist:
                            self.frontier[tuple(neighbor)] = dist
        return

    def solve_it(self, st):
        start_time = time.time()
        
        if self.algorithm == "bfs":
            final_node = self.uninformed_search_bfs(st)
        elif self.algorithm == "dfs":
            final_node = self.uninformed_search_dfs(st)
        else:
            final_node = self.ast_search(st)

        end_time = time.time()
        rusage_denom = 1024.
        if sys.platform == 'darwin':
            rusage_denom = rusage_denom * rusage_denom
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rusage_denom / 100        
            
        moves = deque()
        moves.append(final_node.move)
        current = final_node.parent
        cost_of_path = 0
        while current:
            cost_of_path += 1
            if current.parent <> None:
                moves.append(current.move)
            current = current.parent
        lst_moves = []
        while moves:
            mv = moves.pop()
            if mv == -self.nsize:
                lst_moves.append('Up')
            elif mv == self.nsize:
                lst_moves.append('Down')
            elif mv == -1:
                lst_moves.append('Left')
            else:
                lst_moves.append('Right')
                    
        f = open('output.txt', 'w')
        f.write("path_to_goal: " + str(lst_moves) + "\n")
        f.write("cost_of_path: " + str(cost_of_path) + "\n")
        f.write("nodes_expanded: " + str(self.nodes_expanded) + "\n")
        #f.write("fringe_size: " + str(self.frontier.qsize()) + "\n")
        #f.write("max_fringe_size: " + str(self.max_fringe_size) + "\n")
        f.write("search_depth: " + str(final_node.depth) + "\n")
        f.write("max_search_depth: " + str(self.max_search_depth) + "\n") 
        f.write(("running_time: %s" % (end_time- start_time)) + "\n")
        f.write("max_ram_usage: " + str(mem) + "\n")
        f.close()
        
        if _DEBUG_LEVEL > 0:
            f = open('output.txt', 'r')
            for line in f:
                print line,
            f.close()
        return
            
def main(script, *args):
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        print "Error in arguments!"
        sys.exit()
        
    state = State(3, sys.argv[1])

    st=str(sys.argv[2].split()).replace("'", "")
    start = []
    i = 0
    for c in st:
        if (i % 2) != 0:
            start.append(ord(c) - 48)
        i += 1

    state.solve_it(start)
            
if __name__ == '__main__':
    main(*sys.argv)
    
