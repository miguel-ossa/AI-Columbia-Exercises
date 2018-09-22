#!/usr/bin/python

"""
  Implementation of the IA for the Sudoku problem,
  using AC-3 and Backtracking

  Miguel de la Ossa July, 2017

python -m cProfile -s tottime

"""
import sys
import copy
from collections import deque, OrderedDict
from heapq import heapify, heappush, heappop
import time
#import ujson

_DEBUG_LEVEL = 0

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

class csp:
    def __init__(self, initial_board):
        def constraint(i, j): return (i != j)
        self.X = {} # variables
        self.D = {} # dominios
        self.arcs = {} # arcos
        self.constraint = constraint # obligaciones

        # generate X, D & arcs
        i = 1
        c = 'A'
        for x, value in enumerate(initial_board):
            square = c + str(i)
            self.X[square] = int(value)
            self.arcs[square] = self.generate_arcs(square)
            if value == '0':
                self.D[square] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            else:
                self.D[square] = [int(value)]
            i += 1
            if i > 9:
                i = 1
                c = chr(ord(c) + 1)

        sys.setrecursionlimit(100000)
        return

    def generate_arcs(self, variable):
        arcs = set()
        letter = variable[0]
        n = int(variable[1])

        # column
        temp = set()
        for i in range(1,10): # --> 1..9
            if i != n:
                temp.add(letter + str(i))
        arcs.add(tuple(temp))

        #row
        c = 'A'
        temp = set()
        for alpha in range(0,9): # --> 0..8
            s = chr(ord(c) + alpha) + str(n)
            if s != variable:
                temp.add(s)
        arcs.add(tuple(temp))

        # squares
        if n / 3.0 <= 1:
            rows = [1, 2, 3]
        elif n / 3.0 <= 2:
            rows = [4, 5, 6]
        else:
            rows = [7, 8, 9]

        n_letter = ord(letter) - 64
        if n_letter / 3.0 <= 1:
            columns = ['A', 'B', 'C']
        elif n_letter / 3.0 <= 2:
            columns = ['D' , 'E', 'F']
        else:
            columns = ['G', 'H', 'I']

        temp = set()
        for column in columns:
            for row in rows:
                square = column + str(row)
                if square != variable:
                    temp.add(square)
        arcs.add(tuple(temp))

        return arcs


class Sudoku:
    def __init__(self, csp): #, movs=[]):
        self.csp = csp
        self.queue = deque()
        for kw in self.csp.arcs.items():
            arc = kw[0]
            for k in self.csp.arcs[arc]:
                for neighbor in k:
                    self.queue.append((arc, neighbor))
        return

    def ac3_search(self, csp):
        while self.queue:
            Xi, Xj = self.queue.popleft()
            if self.revise(csp, Xi, Xj):
                if len(csp.D[Xi]) == 0:
                    return False
                for k in csp.arcs[Xi]:
                    for Xk in k:
                        if Xk != Xj:
                            self.queue.append((Xk, Xi))
        return True

    def revise(self, csp, Xi, Xj):
        revised = False
        if len(csp.D[Xj]) > 1:
            return False
        for x in csp.D[Xi]:
            for y in csp.D[Xj]:
                if not csp.constraint(x, y): 
                    csp.D[Xi].remove(x)
                    revised = True
        return revised

    def is_solved(self, csp):
        for k, v in sorted(csp.D.items()):
            if len(csp.D[k]) > 1:
                return False
            for arc in csp.arcs[k]:
                for w in arc:
                    if len(csp.D[w]) > 1:
                        return False
                    if not csp.constraint(csp.D[k], csp.D[w]):
                        return False
        return True

    def print_result(self, csp, method):
        f = open("output.txt", 'w')
        c = 'A'
        for alpha in range(0,9): # --> 0..8
            s = chr(ord(c) + alpha)
            for i in range(1,10): # --> 1..9
                k = s + str(i)
                for v in csp.D[k]:
                    f.write(str(v))
                    #break
        f.write(" " + method)
        f.close()
        return


    #**********************************************************************************************
    #**********************************************************************************************
    def ac3(self, csp, queue):
        while queue:
            Xi, Xj = queue.popleft()
            if self.revise_ac3(csp, Xi, Xj):
                if len(csp.D[Xi]) == 0:
                    return False
                for arc in csp.arcs[Xi]:
                    for Xk in arc:
                        if Xk != Xj:
                            queue.append((Xk, Xi))
        return True

    def revise_ac3(self, csp, Xi, Xj):
        revised = False
        if len(csp.D[Xj]) > 1:
            return False
        for x in csp.D[Xi]: # valores para Di
            #if not self.csp.constraint(self.csp.D[i], self.csp.D[j]):
            for y in csp.D[Xj]:
                if not csp.constraint(x, int(y)):
                    csp.D[Xi].remove(x)
                    revised = True
        return revised

    def select_unassigned_variable(self, csp):
        """
        Se trata de escoger un dominio con el menor numero de elementos (MRV), excluyendo 1 solo elemento.
        Con la siguiente instruccion aparecen primero los de menor numero de elementos, pero los que solamente
        tienen 1 son desplazados al final.
        Se supone que nunca se llamara a esta funcion si todos los dominios tienen un solo valor
        Tambien se supone que csp.D se ira actualizando con assignment
        """
        orderedD = sorted(csp.D, key=lambda k: len(csp.D[k]) if len(csp.D[k]) > 1 else 10, reverse=False)
        return orderedD[0]

    def order_domain_values(self, var, assignment, csp):
        """
        Decidimos el orden en que se examinan los valores de la variable "var", utilizando la
        heuristica "least-constraining-value". Esto es, damos preferencia a los valores que dejan
        mas opciones posibles en las variables del arco.
        """
        lcv = priority_dict()
        for value in csp.D[var]:
            count = 0
            for arc in self.csp.arcs[var]:
                for Xi in arc:
                    if value in csp.D[Xi]:
                        count += 1
                    #for i in csp.D[Xi]:
                    #    if i == value:
                    #        count += 1
            lcv[str(value)] = count
        
        lcvs = set()
        while lcv:
            lcvs.add(int(lcv.pop_smallest()))
        return lcvs

    def inference(self, csp, var, value):
        queue = deque()
        inferences = {}
        for arc in csp.arcs[var]:
            for Xi in arc:
                #inferences[Xi] = 0
                if len(csp.D[Xi]) != 1:
                    queue.append((Xi, var))
                    inferences[Xi] = 0
                    if _DEBUG_LEVEL == 2:
                        print (Xi, csp.D[Xi]),
        csp.D[var] = [value]
        if self.ac3(csp, queue):
            #if _DEBUG_LEVEL == 2:
            #    for k in self.csp.arcs[var]:
            #        for neighbor in k:
            #            if csp.D[neighbor] != Dbak[neighbor]:
            #                print "D changed"
            #if self.arc_is_solved(csp.arcs[var], csp):
            for arc in self.csp.arcs[var]:
                for Xi in arc:
                    if len(csp.D[Xi]) == 1:
                        inferences[Xi] = csp.D[Xi][0]
            return inferences

        return None

    def value_is_consistent_with_assignment(self, csp, assignment, var, value):
        for arc in csp.arcs[var]:
            for Xi in arc:
                if value == assignment[Xi]:
                    return False
        return True
    
    def deepish_copy(self, org):
        '''
        much, much faster than deepcopy, for a dict of the simple python types.
        '''
        out = dict().fromkeys(org)
        for k,v in org.iteritems():
            try:
                out[k] = v.copy()   # dicts, sets
            except AttributeError:
                try:
                    out[k] = v[:]   # lists, tuples, strings, unicode
                except TypeError:
                    out[k] = v      # ints
     
        return out
        
    def backtrack(self, assignment, csp):
        if not 0 in assignment.values():
            return assignment
        var = self.select_unassigned_variable(csp) # Minimum Remaining Value (MRV)
        for value in self.order_domain_values(var, assignment, csp): # least-constraining-value
            assignment_bak = copy.copy(assignment)
            #Dbak = copy.deepcopy(csp.D)
            #Dbak = ujson.loads(ujson.dumps(csp.D))
            Dbak = self.deepish_copy(csp.D)
            
            if self.value_is_consistent_with_assignment(csp, assignment, var, value):
                assignment[var] = value
                inferences = self.inference(csp, var, value)
                if inferences != None:
                    for arc in csp.arcs[var]:
                        for Xi in arc:
                            assignment[Xi] = inferences[Xi]
                    result = self.backtrack(assignment, csp)
                    if result != None:
                        return result
            # remove var = value and inferences from assignment
            assignment = copy.copy(assignment_bak)
            #csp.D = copy.deepcopy(Dbak)
            #csp.D = ujson.loads(ujson.dumps(Dbak))
            for Xi in csp.D:
                if csp.D[Xi] != Dbak[Xi]:
                    csp.D[Xi] = Dbak[Xi]
            #csp.D = self.deepish_copy(Dbak)
        return None

    def backtracking_search(self, csp):
        assignment = copy.copy(csp.X)

        return (self.backtrack(assignment, csp))

def main(script, *args):
    if len(sys.argv) != 2:
        print "Error in arguments!"
        sys.exit()

    initialTime = time.clock()

    myCsp = csp(sys.argv[1])
    #print sys.argv[1]
    mySudoku = Sudoku(myCsp)

    if _DEBUG_LEVEL == 1:
        print ('initial board')
        c = 'A'
        j = 1
        for alpha in range(0,9): # --> 0..8
            for i in range(1,10):
                s = chr(ord(c) + alpha) + str(i)
                print (mySudoku.csp.X[s]),
                j += 1
                if j > 9:
                    j = 1
                    print
        print

    if mySudoku.ac3_search(mySudoku.csp) and mySudoku.is_solved(mySudoku.csp):
            mySudoku.print_result(mySudoku.csp, "AC3\n")
            if _DEBUG_LEVEL == 1:
                print "success!"
    else:
        myCsp = csp(sys.argv[1])
        mySudoku = Sudoku(myCsp)
        assignment = mySudoku.backtracking_search(mySudoku.csp)
        if assignment != None:
            mySudoku.print_result(mySudoku.csp, "BTS\n")
            if _DEBUG_LEVEL == 1:
                c = 'A'
                j = 1
                for alpha in range(0,9): # --> 0..8
                    for i in range(1,10):
                        s = chr(ord(c) + alpha) + str(i)
                        print (assignment[s]),
                        j += 1
                        if j > 9:
                            j = 1
                            print
        if _DEBUG_LEVEL == 1:
            if assignment == None:
                print ('backtracking failed')
            
    
    finalTime = time.clock()
    if _DEBUG_LEVEL == 1:
        print ('time', finalTime - initialTime)
    
    """
    Grading
[Executed at: Fri Jul 7 23:33:13 PDT 2017]

003020600900305001001806400008102900700000008006708200002609500800203009005010300 [5/5]
500400070600090520003025400000000067000014800800600000005200000300007000790350000 [5/5]
000000000047056000908400061000070090409000100000009080000000007000284600302690005 [5/5]
000000000000530041600412005900000160040600002005200000000103200000005089070080006 [5/5]
090060000004000006003000942000200000086000200007081694700008000009510000050000073 [5/5]
000070360301000000042000008003006400004800002000003100005080007200760000000300856 [5/5]
000102900103097000009000070034060800000004500500021030000400000950000000000015307 [5/5]
800000090075209080040500100003080600000300070280005000000004000010027030060900020 [5/5]
000002008401006007002107903007000000065040009004000560000001000008000006910080070 [5/5]
006029000400006002090000600200005104000000080850010263000092040510000000000400800 [5/5]
000000000010720000700014826000000000006000900041906030050001000020097680000580009 [5/5]
005100026230009000000000000000900800590083000006500107060000001004000008853001600 [5/5]
680400000000710009013000000800000300000804090462009000000900037020007108000000026 [5/5]
000900007020007061300810002000078009007300020100040000000000050005000003010052078 [5/5]
000000060000130907900200031002000000004501703010006004046000020000010000200605008 [5/5]
000000000000002891080030507000000000047001085006427003000000000030005070719000204 [5/5]
010050000362010005070206400000005070005090600900000000700001008000374900601000000 [5/5]
000001086076300502000009300007000060900000800054000207008035900030900000000407000 [5/5]
307009000000003060006400001003100094025040803060300002000000006000200900580000040 [5/5]
021000050000000708000400020000600035060000000083020600059002086030001000006904200 [5/5]

003020600900305001001806400008102900700000008006708200002609500800203009005010300
500400070600090520003025400000000067000014800800600000005200000300007000790350000
000000000047056000908400061000070090409000100000009080000000007000284600302690005
000000000000530041600412005900000160040600002005200000000103200000005089070080006
090060000004000006003000942000200000086000200007081694700008000009510000050000073
000070360301000000042000008003006400004800002000003100005080007200760000000300856
000102900103097000009000070034060800000004500500021030000400000950000000000015307
800000090075209080040500100003080600000300070280005000000004000010027030060900020
000002008401006007002107903007000000065040009004000560000001000008000006910080070
006029000400006002090000600200005104000000080850010263000092040510000000000400800
000000000010720000700014826000000000006000900041906030050001000020097680000580009
005100026230009000000000000000900800590083000006500107060000001004000008853001600
680400000000710009013000000800000300000804090462009000000900037020007108000000026
000900007020007061300810002000078009007300020100040000000000050005000003010052078
000000060000130907900200031002000000004501703010006004046000020000010000200605008
000000000000002891080030507000000000047001085006427003000000000030005070719000204
010050000362010005070206400000005070005090600900000000700001008000374900601000000
000001086076300502000009300007000060900000800054000207008035900030900000000407000
307009000000003060006400001003100094025040803060300002000000006000200900580000040
021000050000000708000400020000600035060000000083020600059002086030001000006904200     
    """

    """
    order_domain_values corregido
    duracion de 500068000000000060042050000000800900001000040903000620700001009004200003080000000
                = 90.05460500000001s
                000530000005000600000190503000004000000000164100370800008000040010000008004700921
                = 
    con set() en arcs
    duracion de 500068000000000060042050000000800900001000040903000620700001009004200003080000000
                = 90.846319
                
    con set() tambien en order_domain_values
    duracion de 500068000000000060042050000000800900001000040903000620700001009004200003080000000
                = 100.86725799999999
                
    comprobando assignments con if not 0 in assignments
    duracion de 500068000000000060042050000000800900001000040903000620700001009004200003080000000
                = 92.757962
                
    cambiando deepcopy por copy de assignments
    duracion de 500068000000000060042050000000800900001000040903000620700001009004200003080000000
                = 82.912325
    
    no preguntando por inclusion previa en AC-3            
    duracion de 500068000000000060042050000000800900001000040903000620700001009004200003080000000
                = 47.698087

    eliminado otro not in          
    duracion de 500068000000000060042050000000800900001000040903000620700001009004200003080000000
                = 46.842445999999995
                
    con ujson en lugar de deepcopy!!!         
    duracion de 500068000000000060042050000000800900001000040903000620700001009004200003080000000
                = 14.374848
    
    con deepish_copy!!!
    duracion de 500068000000000060042050000000800900001000040903000620700001009004200003080000000
                = 19.034572999999998
    
    duracion de 094000130000000000000076002080010000032000000000200060000050400000008007006304008
                = 434.200
                
    eliminando 2do deepish_copy
    duracion de 094000130000000000000076002080010000032000000000200060000050400000008007006304008
                = 168.053
    
    
    """


    """
    solved AC3    : 000260701680070090190004500820100040004602900050003028009300074040050036703018000
    solved BTS    : 500068000000000060042050000000800900001000040903000620700001009004200003080000000
                    597468132318927564642153897456832971821796345973514628735641289164289753289375416 BTS 19.034572999999998
                    000530000005000600000190503000004000000000164100370800008000040010000008004700921
                    691538472835247619427196583589614237273859164146372895758921346912463758364785921 BTS 497.850489
                    000000009024059003750000060000000000070030890000065742800002000000006010043008000
                    318624579624759183759183264286947351475231896931865742867412935592376418143598627 BTS 21.405585000000002
                    040012050000300006008007009000000103317000000400009080070000000000800000090620805
                    746912358921358746538467219689745123317286594452139687873591462265874931194623875 BTS 30.83612
                    094000130000000000000076002080010000032000000000200060000050400000008007006304008  
                    794582136268931745315476982689715324432869571157243869821657493943128657576394218 BTS 434.200
    """

    """
mossa$ python -m cProfile -s tottime driver.py 094000130000000000000076002080010000032000000000200060000050400000008007006304008

          381253340 function calls (380878030 primitive calls) in 434.200 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
  2027749  233.035    0.000  239.667    0.000 driver.py:336(deepish_copy)
   406607   48.894    0.000  100.362    0.000 driver.py:244(ac3)
 59561679   34.786    0.000   42.482    0.000 driver.py:256(revise_ac3)
 375311/1   31.901    0.000  434.162  434.162 driver.py:352(backtrack)
   375311   12.250    0.000   26.642    0.000 {sorted}
 30400110   11.409    0.000   14.392    0.000 driver.py:276(<lambda>)
123854134    9.549    0.000    9.549    0.000 {len}
   406607    8.772    0.000  110.994    0.000 driver.py:302(inference)
   375310    7.841    0.000   13.371    0.000 driver.py:279(order_domain_values)
  2027749    6.319    0.000    6.319    0.000 {built-in method fromkeys}
  2027750    6.302    0.000    6.302    0.000 copy.py:113(_copy_with_constructor)
 63083274    4.754    0.000    4.754    0.000 {method 'append' of 'collections.deque' objects}
 59570179    4.380    0.000    4.380    0.000 {method 'popleft' of 'collections.deque' objects}
 19696339    2.407    0.000    2.407    0.000 driver.py:108(constraint)
  2027750    1.995    0.000    8.602    0.000 copy.py:66(copy)
  1013901    1.952    0.000    1.952    0.000 driver.py:329(value_is_consistent_with_assignment)
  1013971    1.691    0.000    2.274    0.000 driver.py:70(__setitem__)
  1013971    1.319    0.000    1.715    0.000 driver.py:57(pop_smallest)
   375310    0.724    0.000    1.376    0.000 driver.py:36(__init__)
  2629409    0.621    0.000    0.621    0.000 {method 'remove' of 'list' objects}
   375310    0.583    0.000   27.225    0.000 driver.py:268(select_unassigned_variable)
   375310    0.530    0.000    0.652    0.000 driver.py:40(_rebuild_heap)
   375311    0.450    0.000    0.450    0.000 {method 'values' of 'dict' objects}
  1013971    0.402    0.000    0.402    0.000 {_heapq.heappush}
  1013971    0.396    0.000    0.396    0.000 {_heapq.heappop}
  2403059    0.382    0.000    0.382    0.000 {method 'iteritems' of 'dict' objects}
  2027750    0.305    0.000    0.305    0.000 {method 'get' of 'dict' objects}
  1018345    0.166    0.000    0.166    0.000 {method 'add' of 'set' objects}
   375310    0.053    0.000    0.053    0.000 {_heapq.heapify}
        1    0.012    0.012    0.022    0.022 driver.py:192(ac3_search)
     8500    0.006    0.000    0.008    0.000 driver.py:204(revise)
      162    0.005    0.000    0.007    0.000 driver.py:133(generate_arcs)
        2    0.002    0.001    0.003    0.001 driver.py:182(__init__)
        1    0.002    0.002    0.003    0.003 collections.py:1(<module>)
        1    0.001    0.001    0.001    0.001 heapq.py:31(<module>)
        1    0.001    0.001  434.200  434.200 driver.py:11(<module>)
        1    0.001    0.001    0.001    0.001 {open}
        1    0.001    0.001  434.196  434.196 driver.py:383(main)
        2    0.001    0.000    0.007    0.004 driver.py:107(__init__)
     1647    0.000    0.000    0.000    0.000 {chr}
      354    0.000    0.000    0.000    0.000 {range}
     1809    0.000    0.000    0.000    0.000 {ord}
        3    0.000    0.000    0.000    0.000 {method 'items' of 'dict' objects}
        1    0.000    0.000    0.001    0.001 driver.py:227(print_result)
        1    0.000    0.000    0.000    0.000 {method 'close' of 'file' objects}
        1    0.000    0.000    0.000    0.000 keyword.py:11(<module>)
        1    0.000    0.000  434.162  434.162 driver.py:378(backtracking_search)
        1    0.000    0.000    0.000    0.000 driver.py:215(is_solved)
       82    0.000    0.000    0.000    0.000 {method 'write' of 'file' objects}
        1    0.000    0.000    0.000    0.000 collections.py:26(OrderedDict)
        2    0.000    0.000    0.000    0.000 {time.clock}
        1    0.000    0.000    0.000    0.000 driver.py:181(Sudoku)
        1    0.000    0.000    0.000    0.000 collections.py:395(Counter)
        1    0.000    0.000    0.000    0.000 driver.py:21(priority_dict)
        2    0.000    0.000    0.000    0.000 {sys.setrecursionlimit}
        1    0.000    0.000    0.000    0.000 driver.py:106(csp)
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
    """

if __name__ == '__main__':
    main(*sys.argv)

