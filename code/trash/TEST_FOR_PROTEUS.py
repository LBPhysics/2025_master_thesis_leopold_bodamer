import numpy as np
import time

# Function to run a computationally intensive task (Matrix Multiplication)
def compute_intensive_task():
    # Create two large random matrices
    size = 10000  # You can adjust this size based on available memory and processing power
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)

    # Start the timer
    start_time = time.time()

    # Perform matrix multiplication
    C = np.dot(A, B)

    # Stop the timer
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time

    return elapsed_time

# Run the task and measure the time
elapsed_time = compute_intensive_task()

print(f"Task completed in: {elapsed_time:.4f} seconds")