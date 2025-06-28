import time
from abc import ABC,abstractmethod
from queue import Queue

class Player(ABC):
    def __init__(self,name:str):
        self.name=name
        self.start_time=None
        self.end_time=None

    def start_timer(self):
        self.start_time=time.time()

    def stop_timer(self):
        self.end_time=time.time()

    def get_duration(self):
        if self.start_time is not None and self.end_time is not None:
            return self.end_time-self.start_time
        else:
            return None
        
    @abstractmethod
    def playMethod(self):
        pass    


class HumanPlayer(Player):
    ...
    def playMethod(self):
        ...

class RobotPlayer(Player):
    def __init__(self,name:str):
        super().__init__(name)
        self.maze=[
            [" ", "#", "#", "#", " ", "#", "#", "#", "#", "#"],
            ["#", " ", "#", " ", " ", " ", " ", "#", " ", "#"],
            ["#", " ", "#", " ", "#", " ", " ", "#", " ", "#"],
            ["#", " ", " ", " ", "#", " ", " ", "#", " ", "#"],
            ["#", " ", " ", " ", "#", " ", " ", "#", " ", "#"],
            ["#", " ", "#", " ", " ", " ", " ", " ", " ", "#"],
            ["#", " ", "#", " ", "#", " ", " ", " ", " ", "#"],
            ["#", " ", " ", " ", "#", "#", " ", " ", " ", "#"],
            ["#", " ", " ", " ", " ", " ", " ", " ", " ", "#"],
            ["#", "#", "#", "#", "#", "#", "#", "#", "#", " "]
                  ]
        self.path=[]

    def display_maze(self):
        maze_=''
        for row in self.maze:
            for tile in row:
                maze_ +=' '+tile
            print(maze_)
            maze_=''

    def transform_to_graph(self):
        graph={}
        rows,cols=len(self.maze),len(self.maze[0])
        for row in range(rows):
            for col in range(cols):
                if self.maze[row][col]!='#':
                    adj=[]
                    if row+1<rows and self.maze[row+1][col]!='#':
                        adj.append((row+1,col))
                    if row-1>=0 and self.maze[row-1][col]!='#':
                        adj.append((row-1,col))
                    if col+1<cols and self.maze[row][col+1]!='#':
                        adj.append((row,col+1))
                    if col-1>=0 and self.maze[row][col-1]!='#':
                        adj.append((row,col-1))
                    graph[(row,col)]=adj
        return graph           

    def solve_maze_bfs(self,graph,start,end):
        visited=[]
        queue=Queue()
        queue.put([start])

        while not queue.empty():
            path=queue.get()
            neighbours=path[-1]

            if neighbours==end:
                for row,col in path:
                    if self.maze[row][col]==' ':
                        self.maze[row][col]='p'
                return path
            for neighbor in graph.get(neighbours,[]):
                if neighbor not in visited:
                    visited.append(neighbor)
                    queue.put(path+[neighbor])
        return []            

    def playMethod(self):
        ...
  
