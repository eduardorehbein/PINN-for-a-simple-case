from pinn import HighLevelRLCircuitPINN

# PINN instancing
R = 3
L = 3
hidden_layers = [9, 9]
learning_rate = 0.001

subpinns = list()

# Train subpinns with v = [-10, -6, -2, 2, 6, 10]
# Train high level pinn with v = [-8, -4, 0, 4, 8]
# Test high level pinn with v = [-9, -7, -5, -3, -1, 1, 3, 5, 7, 9]

high_level_model = HighLevelRLCircuitPINN(R, L, subpinns, hidden_layers, learning_rate)

