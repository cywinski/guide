from argparse import Namespace


class CommWorld:
    def __init__(self):
        self.rank = 0
        self.size = 1

    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def bcast(self, value, root=0):
        return value


MPI = Namespace(COMM_WORLD=CommWorld())
