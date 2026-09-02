from collections import defaultdict


points = [[1,2,3], 
          [4,5,6],
          [7,8,9],
          [9,8,7],
          [6,5,4],
          [3,2,1],
          [21,32,43],
          [54,65,76],
          [87,98,12],
          [1,2,5],
          [3,5,6],
          [1,5,7],
          [7,8,9]]


class Graphs:
    def __init__(self) :
        self.nodes = {}
        self.len_node = 0
        self.edge = defaultdict(dict)

    def add_node(self, position):
        self.len_node = len(self.nodes)
        self.nodes[self.len_node] = [True, position]

    def delete_node(self, index:int):

        self.nodes[index][0] = False
        self.nodes[index][1] = []

    def add_edge(self,a, b):
        if self.nodes[a][0] and self.nodes[b][0] :
            self.edge[a][b] = 0
            self.edge[b][a] = 0
        else :
            print("failed to make edge")



if __name__ == "__main__":
    map = Graphs()

    

    for point in points :

        map.add_node(point)

    map.delete_node(2)

    map.add_edge(0,1)
    map.add_edge(3,4)

    
    for id, node in map.nodes.items() :
        if node[0] :
            print(id, node[1])

    print(map.edge)





