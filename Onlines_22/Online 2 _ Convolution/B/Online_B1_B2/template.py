import numpy as np
import matplotlib.pyplot as plt

# Todo: Define Signal class

# Todo: Define LTI class

if __name__ == "__main__":
    INF = 10

    x = Signal(INF)
    x.set_value_at_time(0, 1)
    x.set_value_at_time(2, -1)
    x.plot("Input x(n)")

    h1 = Signal(INF)
    h1.set_value_at_time(0, 1)

    h2 = Signal(INF)
    h2.set_value_at_time(1, 0.5)

    h3 = Signal(INF)
    h3.set_value_at_time(0, 1)
    h3.set_value_at_time(1, 1)

    sys1 = LTI_System(h1)
    sys2 = LTI_System(h2)
    sys3 = LTI_System(h3)
    
    # Todo: Determine output block by block

    y_final_1.plot("Output via block-by-block system")

    # Todo: Determine h_combined
    
    sys_combined = LTI_System(h_combined)

    y_final_2 = sys_combined.output(x)
    y_final_2.plot("Output via combined impulse response")

    print("Outputs are equal:",
          np.allclose(y_final_1.values, y_final_2.values))
