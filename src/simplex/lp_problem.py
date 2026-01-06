
class LpProblem:
    
    def __init__(self, A, b, c):
        self.A_ = A
        self.b_ = b
        self.c_ = c
    
    @property
    def constraint_matrix(self):
        return self.A_
    
    @property
    def rhs(self):
        return self.b_
    
    @property
    def objective(self):
        return self.c_
    
    @property
    def variables(self):
        return len(self.c_)