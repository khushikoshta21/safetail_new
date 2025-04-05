import random
import pandas as pd
import numpy as np
from csv import writer
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import LabelEncoder


discount_factor = 0.95
req_satisfaction_count = 0
total_steps = 3
total_episodes = 10
no_of_es = 10
epsilon = 1
time_step = 0
time_episode = 0
action_space = {}
step_reward_array = []
step_array = []
episode_reward_array = []
episode_array = []

df = pd.read_csv("/home/user/khushi/SafeTail 2.0/datasets/multi_cloud_service_dataset.csv")
encoder = LabelEncoder()
df['Service_Type'] = encoder.fit_transform(df['Service_Type'])



#print(df['Service_Type'])

#selecting action
def get_action(no_of_es,step,selected_tasks,mlp_model,scaler):

    global df2
    global epsilon
    action_space = {}

    # Ensure step index is within range
    if step >= len(df):
        print(f"Warning: Step {step} is out of range. Using last available step.")
        step = len(df2) - 1

    #Ensure only the required columns are used
    expected_features = scaler.feature_names  # Get the exact features used during training
    df_filtered = df[expected_features]  # Keep only relevant features

    # Extract the state for the given step
    state_flattened = df_filtered.iloc[step].values.reshape(1, -1)

    # Normalize the input using the saved scaler
    state_flattened = scaler.transform(state_flattened)

    if epsilon >= 1.5:  # Exploration (random selection)
        selected_edge_servers = generate_subsets()
    else:  # Exploitation (MLP Regressor)
        print(f"State shape before predict: {state_flattened.shape}")
        predicted_wt = mlp_model.predict(state_flattened)[0]  # Predict waiting times
        
        # Select best servers (lowest waiting time)
        sorted_servers = np.argsort(predicted_wt)
        selected_edge_servers = sorted_servers[:random.randint(1, 4)].tolist()
        
        print("Using MLP Regressor for action selection")

    #for task in selected_tasks:
        #action_space[task] = selected_edge_servers

    return selected_edge_servers




#req satisfaction check
def get_sat_service_req_count(tasks,selected_edge_servers,step):
    isSatisfied = False
    req_satisfaction_count = 0

    #calculating waiting time
    waiting_time = float(get_waiting_time(selected_edge_servers,step))
    
    execution_time = float(df3.loc[df3['Combination'] == tasks, 'Execution Time (seconds)'].mean())

    if( waiting_time + (execution_time/100) < execution_time):
        isSatisfied = True
            
    if isSatisfied:
        req_satisfaction_count += 1

    return req_satisfaction_count


#episodic reward
def get_episodic_reward(discount_factor, total_steps,selected_edge_servers,selected_tasks,tasks,sat_service_req_count_per_episode,avg_wt):
    Gt_return = 0
    
    for step in range(0,total_steps):
        step_reward = get_step_reward(selected_edge_servers,step)
        Gt_return += pow(discount_factor,step) * pow(step_reward,step+1)

    total_req_recieved = len(selected_tasks)
    total_req_satisfied = sat_service_req_count_per_episode

    deg_of_satisfaction = total_req_satisfied / total_req_recieved

    episodic_reward = Gt_return + deg_of_satisfaction - avg_wt

    print("Episodic Reward: ", episodic_reward)

    return episodic_reward
 

#step reward
def get_step_reward(selected_edge_servers,step):
    queue_length = 0
    total_queue_length = 0
    for server in range (1,11):
        total_queue_length += df2[f"Requests_In_Queue_Server_{server}"].iloc[step]

    for e in selected_edge_servers:
        queue_length_edge = df2[f"Requests_In_Queue_Server_{e}"].iloc[step]
        queue_length += (queue_length_edge - (total_queue_length/10))

    step_reward = queue_length - get_min_wt(selected_edge_servers,step)

    #print(f"Reward at step {step} is {step_reward}")
    return step_reward


#minimum waiting time
def get_min_wt(selected_edge_servers,step):
    min_wt = 888888888888
    for e in selected_edge_servers:
        arrival_rate = float(df2.at[step,f"Arrival_Rate_Server_{e}"])
        service_rate = float(df2.at[step,f"Service_Rate_Server_{e}"])
        wt = ( arrival_rate / service_rate * (service_rate - arrival_rate))

        #print(f"Waiting time for {e} is {wt}")
        
        df2.at[step, f"Waiting_Time_Server_{e}"] = wt

        #changing arrival and service rates
        df2.at[step,f"Arrival_Rate_Server_{e}"] += random.randint(0,5)
        df2.at[step,f"Service_Rate_Server_{e}"] += random.randint(0,5)

        min_wt = min(wt,min_wt)

        return min_wt


#only waiting time
def get_waiting_time(selected_edge_servers,step):
    for e in selected_edge_servers:
        arrival_rate = float(df2.at[step,f"Arrival_Rate_Server_{e}"])
        service_rate = float(df2.at[step,f"Service_Rate_Server_{e}"])
        wt = ( arrival_rate / service_rate * (service_rate - arrival_rate))

        #print(f"Waiting time for {e} is {wt}")
    return wt

#task selection

def selecting_tasks(num_of_tasks):
    
    tasks = ["predict", "speech", "detect"]
    
    service_requests = []
    for i in range (num_of_tasks):

        task_index = random.randint(0,2)
        service_requests.append(tasks[task_index])
    
    return service_requests


#generating subsets
edge_servers = [1,2,3,4,5,6,7,8,9,10] #assuming 0 is controller

def generate_subsets():
    #selecting edge servers --> action
    num_of_edge_ser_selected = random.randint(1,4)

    edge_ser_sel = []
    while(len(edge_ser_sel) != num_of_edge_ser_selected):
            
        edge = random.randint(1,10)
        if(edge not in edge_ser_sel):
            edge_ser_sel.append(edge)

    return edge_ser_sel


def controller():
    
        # Correctly load the trained MLP model
    try:
        with open("/home/user/khushi/SafeTail 2.0/Multicloud/multicloud_model.pkl", "rb") as model_file:  # Change name if needed
            mlp_model = pickle.load(model_file)
        print("MLP model loaded successfully!")
    except Exception as e:
        print(f"Error loading MLP model: {e}")

    # Correctly load the scaler
    try:
        with open("/home/user/khushi/SafeTail 2.0/Multicloud/multicloud_scaler.pkl", "rb") as f:  # Change name if needed
            scaler = pickle.load(f)
        print("Scaler loaded successfully!")
    except Exception as e:
        print(f"Error loading scaler: {e}")

    # Ensure scaler has feature names
    if not hasattr(scaler, 'feature_names'):
        raise ValueError("Scaler does not have feature names. Retrain the model and save it properly.")

    feature_names = scaler.feature_names  # Get the correct feature list
    
    
    #episode
    for ep in range(1,total_episodes):

        #episode_array.append((ep+=total_steps))
        global time_episode
        time_episode += total_steps

        episode_array.append(time_episode)

        #steps
        for step in range(1,total_steps):

            global time_step
            step_array.append(time_step)
            time_step += 1

            #copying the old state into new row
            df['index_col'] = df.index
            last_index = df.iloc[-1]['index_col']
            print(last_index)

            df.loc[last_index + step - 1] = df.iloc[last_index + step - 2].copy()

            print(f"Step {step} completed")
            

        print(df.head())
        print(df.tail())
        #episodic reward    
        
        
        #episodic_reward = get_episodic_reward(discount_factor, total_steps,selected_edge_servers,selected_tasks,tasks,sat_service_req_count_per_episode,avg_wt)
        
        #print (f"Episode {ep} completed")
        #print()
        #print("------------------------------------------------")
        #print()

        #episode_reward_array.append(episodic_reward)
        
        

        #plotting step reward
        #plt.xlabel("Timestep")
        #plt.ylabel("Step Reward")
        #plt.title("Step Reward over timesteps")

        #plt.plot(step_array,step_reward_array,label="Step Reward", color="green")
        #plt.show()

        #df2.to_csv('datasets/cleaned_edge_servers_data.csv')

    #plt.xlabel("Episode")
    #plt.ylabel("Episodic Reward")
    #plt.title("Episodic Reward over episodes")
    #plt.plot(step_array,step_reward_array,label="Step Reward", color="green")
    #plt.plot(episode_array,episode_reward_array,label="Episodic Reward", color="red")
    #print("step array: ", step_array )
    print("-----------------------------------------------------------------------------")
    #print("episode_reward_array: ", episode_reward_array )
    print("-----------------------------------------------------------------------------")
    #print("step_reward_array: ", step_reward_array )
    print("-----------------------------------------------------------------------------")

    
    #plt.show()
    
controller()