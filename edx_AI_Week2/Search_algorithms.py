#!/usr/bin/python -tt
# Copyright 2010 Google Inc.
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0

# Google's Python Class
# http://code.google.com/edu/languages/google-python-class/


def find_path(graph, start, end, path=[]):
  path = path + [start]
  if start == end:
    return path
  if not graph.has_key(start):
    return None
  for node in graph[start]:
    if node not in path:
      newpath = find_path(graph, node, end, path)
      if newpath: return newpath
  return None

def find_all_paths(graph, start, end, path=[]):
  path = path + [start]
  if start == end:
    return [path]
  if not graph.has_key(start):
    return []
  paths = []
  for node in graph[start]:
    if node not in path:
      newpaths = find_all_paths(graph, node, end, path)
      for newpath in newpaths:
        paths.append(newpath)
  return paths

def find_shortest_path(graph, start, end, path=[]):
  path = path + [start]
  if start == end:
    return path
  if not graph.has_key(start):
    return None
  shortest = None
  for node in graph[start]:
    if node not in path:
      newpath = find_shortest_path(graph, node, end, path)
      if newpath:
        if not shortest or len(newpath) < len(shortest):
          shortest = newpath
  return shortest

# Provided main() calls the above functions with interesting inputs,
# using test() to check if each result is correct or not.
def main():
  graph2 = {'S': ['A', 'B', 'C'],
            'A': ['S', 'D'],
            'B': ['S', 'D', 'E', 'G'],
            'C': ['S', 'E'],
            'D': ['A', 'B', 'F'],
            'E': ['B', 'C', 'F', 'H'],
            'F': ['D', 'E', 'G'],
            'G': ['B', 'F', 'H'],
            'H': ['E', 'G']}
  path = find_path(graph2, 'S', 'G')
  print 'path: ' + str(path)

  all_paths = find_all_paths(graph2, 'S', 'G')
  print "all: " + str(all_paths)

  shortest = find_shortest_path(graph2, 'S', 'G')
  print "shortest: " + str(shortest)

# Standard boilerplate to call the main() function.
if __name__ == '__main__':
  main()
