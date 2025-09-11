import numpy as np

# Load the q_table.npy file
q_table_path = './learning_state/q_table.npy'
q_table = np.load(q_table_path)

# Print the last 5 entries
print(f"amount of entries : {len(q_table)}")
print("Last 5 entries of the Q-table:")
print(q_table[-5:])
