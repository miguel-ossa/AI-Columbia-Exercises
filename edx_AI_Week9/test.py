#!/usr/bin/python -tt
# Copyright 2010 Google Inc.
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0

# Google's Python Class
# http://code.google.com/edu/languages/google-python-class/

import driver
import sys
import time
       
def main():
    if len(sys.argv) != 5:
        print "Error in arguments!"
        sys.exit()
        
    with open(sys.argv[1]) as f:
        sudokus_start = f.readlines()
    sudokus_start = [x.strip() for x in sudokus_start] # remove whitespace characters like `\n` at the end of each line
    #print "sudokus_start= ", sudokus_start

    sudokus_finish = []
    with open(sys.argv[2]) as f:
        for line in f:
            strTemp = str(line)
            tempList = []
            for part in strTemp.split():
                tempList.append(part)
            tempList = [x.strip() for x in tempList] # remove whitespace characters like `\n` at the end of each line
            sudokus_finish.append(tempList)
                
    #print "sudokus_finish=", sudokus_finish

    all_output = open(sys.argv[3], 'w')
    all_output_with_times = open(sys.argv[4], 'w')
    # call driver for every input board
    for input_board in sudokus_start:
        initialTime = time.clock()
        myCsp = driver.csp(input_board)
        mySudoku = driver.Sudoku(myCsp)
        if mySudoku.ac3_search(mySudoku.csp) and mySudoku.is_solved(mySudoku.csp):
            finalTime = time.clock()
            mySudoku.print_result(mySudoku.csp, "AC3")
            print (input_board, 'solved with AC3')
            with open("output.txt") as input:
                for line in input:
                    all_output.write(line + "\n")
            all_output.close()
            all_output = open(sys.argv[3], 'a')
            with open("output.txt") as input:
                for line in input:
                    all_output_with_times.write(line + " " + str(finalTime - initialTime) + "\n")
            all_output_with_times.close()
            all_output_with_times = open(sys.argv[4], 'a')
        else:
            myCsp = driver.csp(input_board)
            mySudoku = driver.Sudoku(myCsp)
            assignment = mySudoku.backtracking_search(mySudoku.csp)
            if assignment != None:
                finalTime = time.clock()
                mySudoku.print_result(mySudoku.csp, "BTS")
                print (input_board, 'solved with BTS')
                with open("output.txt") as input:
                    for line in input:
                        all_output.write(line + "\n")
                all_output.close()
                all_output = open(sys.argv[3], 'a')
                with open("output.txt") as input:
                    for line in input:
                        all_output_with_times.write(line + " " + str(finalTime - initialTime) + "\n")
                all_output_with_times.close()
                all_output_with_times = open(sys.argv[4], 'a')
            else:
                print ('backtracking failed')

        
        
    all_output.close()
    all_output_with_times.close()

if __name__ == '__main__':
    main()
