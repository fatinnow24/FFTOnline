import numpy as np
import matplotlib.pyplot as plt

# Todo: Define Signal class
        
class SuperSignal:
    def __init__(self):
        self.components = []

    def add(self, signal: Signal, coefficient=1.0):
        self.components.append((coefficient, signal))
        
# Todo: Define LTI class

if __name__ == "__main__":
    INF = 10

    # Component signals
    x1 = Signal(INF)
    x1.set_value_at_time(0, 1)

    x2 = Signal(INF)
    x2.set_value_at_time(2, 1)

    # Todo: Create SuperSignal: x(n) = 2*x1(n) - x2(n)

    # Impulse response
    h = Signal(INF)
    h.set_value_at_time(0, 1)
    h.set_value_at_time(1, 0.5)

    system = LTI_System(h)

    # Todo: Output using superposition
