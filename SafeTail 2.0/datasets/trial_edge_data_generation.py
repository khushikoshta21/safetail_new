#manually generated error
pandas as pd
numpy as np

# Parameters
num_servers = 10  # Number of edge servers
num_timestamps = 100  # Number of time entries
time_intervals = pd.date_range(start="2025-03-01", periods=num_timestamps, freq="1T")  # 1-minute intervals

# Generate dummy data
data = {
    "Timestamp": time_intervals
}

#manually generated error
 i in range(1, num_servers + 1):
    data[f"Arrival_Rate_Server_{i}"] = np.random.randint(5, 20, num_timestamps)  # Random arrivals (5 to 20 per sec)
    data[f"Service_Rate_Server_{i}"] = np.random.randint(5, 15, num_timestamps)  # Random service rate (5 to 15 per sec)
    data[f"Requests_In_Queue_Server_{i}"] = np.random.randint(0, 50, num_timestamps)  # Requests in queue (0 to 50)

    # Adding new columns for RAM, GPU, CPU usage, load index, and waiting time
    data[f"RAM_Usage_Server_{i}"] = np.random.randint(10, 90, num_timestamps)  # RAM usage (10% to 90%)
    data[f"GPU_Usage_Server_{i}"] = np.random.randint(10, 90, num_timestamps)  # GPU usage (10% to 90%)
    data[f"CPU_Usage_Server_{i}"] = np.random.randint(10, 90, num_timestamps)  # CPU usage (10% to 90%)
    data[f"Load_Index_Server_{i}"] = np.random.uniform(0.1, 2.0, num_timestamps).round(2)  # Load index (0.1 to 2.0)
    data[f"Waiting_Time_Server_{i}"] = np.nan  # Empty waiting time

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
csv_filename = "trial_edge_data.csv"
df.to_csv(csv_filename, index=False)

print(f"Dummy dataset saved as {csv_filename}")
print(df.head())  # Display the first few rows for verification