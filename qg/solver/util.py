class _Math():
    def __init__(self, value):
        self.value = value
        
    def __mul__(self, b):
        return self.value * b
    
    def __rmul__(self, a):
        return a * self.value
    
    def __add__(self, b):
        return self.value + b
    
    def __radd__(self, a):
        return a + self.value
    
    def __sub__(self, b):
        return self.value - b    
    
    def __rsub__(self, a):
        return a - self.value
    
    def __negate__(self):
        return -self.value
    
    def __truediv__(self, b):
        return self.value / b
    
    def __rtrudiv__(self, a):
        return a / self.value
    
    def __pow__(self, b):
        return self.value ** b
    
    def __rpow__(self, a):
        return a ** self.value

class _Cache():
    def __init__(self, level, optable):
        self.level = level
        self.optable = optable
        self.cache = []
    
    def update(self, value):
        self.cache = [value, *self.cache[:self.level-1]]
        
    def __call__(self, value):
        self.update(value)
        return self.optable(self.cache)

def dot(a,b):
    return sum([a[i]*b[i] for i in range(len(a))])

cached_dot = lambda table: (lambda b: dot(table[len(b)],b))