import time
from abc import ABC,abstractmethod

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
    ...
    def playMethod(self):
        ...
  
