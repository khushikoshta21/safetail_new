import random
import pandas as pd
import numpy as np
#from mlp_regressor import predict
#from mlp_regressor import predict
from csv import writer
import matplotlib.pyplot as plt
import pickle
import time
import datetime

discount_factor = 0.95
req_satisfaction_count = 0
total_steps = 5
total_episodes = 3
no_of_es = 10
epsilon = 0.1
time_step = 0
time_episode = 0
action_space = {}
step_reward_array = []
step_array = []
episode_reward_array = []
episode_array = []

df2 = pd.read_csv("/home/user/khushi/SafeTail 2.0/datasets/synthetic_edge_data.csv")
df3 = pd.read_csv("datasets/service_request_unique_sorted_chars.csv")



#selecting action
def get_action(no_of_es,step,selected_tasks,mlp_model,scaler):

    global df2
    global epsilon

    # Ensure step index is within range
    """if step >= len(df2):
        print(f"Warning: Step {step} is out of range. Using last available step.")
        step = len(df2) - 1"""

    #Ensure only the required columns are used
    expected_features = scaler.feature_names  # Get the exact features used during training
    df_filtered = df2[expected_features]  # Keep only relevant features

    # Extract the state for the given step
    state_flattened = df_filtered.iloc[step].values.reshape(1, -1)

    # Normalize the input using the saved scaler
    state_flattened = scaler.transform(state_flattened)

    if random.random() > epsilon:  # Exploration 
        selected_edge_servers = generate_subsets()
        print("Exploration (random selection)")
    else:  
        predicted_wt = mlp_model.predict(state_flattened)[0]  # Predict waiting times

        sorted_servers = [i+1 for i in np.argsort(predicted_wt) if i+1 in range(1, 11)]
        selected_edge_servers = sorted_servers[:random.randint(1, 4)]
        print("Exploitation")
        
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
        wt = abs(( arrival_rate / service_rate * (service_rate - arrival_rate)))

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
        wt = abs(( arrival_rate / service_rate * (service_rate - arrival_rate)))

        #print(f"Waiting time for {e} is {wt}")
    return wt


#task selection
def selecting_tasks(num_of_tasks):
    
    tasks = ["predict", "speech", "detect"]
    
    service_requests = []
    for i in range (num_of_tasks):

        task_index = random.randint(0,2)
        service_requests.append(tasks[task_index])
        print("Selected tasks: ", service_requests)
    
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
        with open("/home/user/khushi/SafeTail 2.0/Regressors/predicting_waiting_time/mlp_multi_output_model.pkl", "rb") as model_file:
            mlp_model = pickle.load(model_file)
        #print("MLP model loaded successfully!")
    except Exception as e:
        print(f"Error loading MLP model: {e}")

    # Correctly load the scaler
    try:
        with open("/home/user/khushi/SafeTail 2.0/Regressors/predicting_waiting_time/scaler_2.pkl", "rb") as f:
            scaler = pickle.load(f)
        #print("Scaler loaded successfully!")
    except Exception as e:
        print(f"Error loading scaler: {e}")

    # Ensure scaler has feature names
    if not hasattr(scaler, 'feature_names'):
        raise ValueError("Scaler does not have feature names. Retrain the model and save it properly.")

    feature_names = scaler.feature_names  # Get the correct feature list
    
    
    #episode
    for ep in range(1,total_episodes):
        
        start_time = time.time()
        global time_episode, df2
        time_episode += total_steps

        episode_array.append(time_episode)

        #steps
        for i in range(1,total_steps):

            global time_step
            step_array.append(time_step)
            time_step += 1
            #print(df2.shape)

            #selecting tasks 
            num_of_tasks = random.randint(1,4)
            
            selected_edge_servers = generate_subsets()
            selected_tasks = selecting_tasks(num_of_tasks)
            for j in range(num_of_tasks):
                action_space[selected_tasks[j]] = get_action(no_of_es,i,selected_tasks,mlp_model,scaler)

            print("action space: " , action_space)

            tasks = ""
            for str in selected_tasks:
                tasks += str[0]
            #print(tasks)

            def sortTask(tasks):
                return ''.join(sorted(tasks))
            
            sorted_tasks = sortTask(tasks)
            print(f" Task for step {i} is {sorted_tasks}")
            
            valid_task = []
            #check if task exists
            for index, row in df3.iterrows():
                valid_task.append(row['Combination'])
            
            #print(valid_task)

            last_index = int(df2.iloc[-1]['Index'])
            #print("last_index_is: " ,last_index)
            #print ("new index is: ", last_index + 1)

            new_row = df2.iloc[-1].copy()  # Copy last row
            new_row["Index"] = last_index + 1  # Update index
            new_row["Timestep"] += 1  # Update timestep
            df2 = pd.concat([df2, pd.DataFrame([new_row])], ignore_index=True)

        
            #print(f"df2 at step {i}: ", df2.loc[last_index+1:])
            
            #changing state of selected servers
            for e in selected_edge_servers:
                
                #change queue length
                df2.at[last_index + 1,f'Requests_In_Queue_Server_{e}'] += 1
                

                #to change from service req --> cpu gpu ram

                if sorted_tasks in valid_task:
                    #cpu utilisation
                    service_cpu_req = df3.loc[df3["Combination"] == sorted_tasks, 'Number of Cores Used']

                    if isinstance(service_cpu_req,pd.Series) and not service_cpu_req.empty:
                        df2.at[last_index + 1,f"CPU_Cores_Usage_Server_{e}"] -= service_cpu_req.iloc[0]
                    else:
                        df2.at[last_index + 1,f"CPU_Cores_Usage_Server_{e}"] -= service_cpu_req


                    #gpu m/m utilisation
                    service_gpu_mm_req = df3.loc[df3["Combination"] == sorted_tasks, 'GPU Memory Usage (MB)']
                    if isinstance(service_gpu_mm_req,pd.Series) and not service_gpu_mm_req.empty:
                        df2.at[last_index + 1,f"GPU_Memory_Usage_Server_{e}"] -= service_gpu_mm_req.iloc[0]
                    else:
                        df2.at[last_index + 1,f"GPU_Memory_Usage_Server_{e}"] -= service_gpu_mm_req
                    

                    # ram utilisation
                    service_ram_req = df3.loc[df3["Combination"] == sorted_tasks, 'RAM Memory Usage (MB)']
                    if isinstance(service_ram_req,pd.Series) and not service_ram_req.empty:
                        df2.at[last_index + 1,f"RAM_Memory_Usage_Server_{e}"] -= service_ram_req.iloc[0]
                    else:
                        df2.at[last_index + 1,f"RAM_Memory_Usage_Server_{e}"] -= service_ram_req
                    
                    
                    #to calculate and change --> load index , arrival rate, service rate and waiting time 
                    df2.at[last_index + 1,f"Load_Index_Server_{e}"] = 0.25 * (float(df2.at[last_index + 1,f"Requests_In_Queue_Server_{e}"])) + (float(df2.at[last_index + 1,f"RAM_Memory_Usage_Server_{e}"]) / float(df2.at[last_index + 1,f"Total_RAM_Memory_Server_{e}"])) + (float(df2.at[last_index + 1,f"GPU_Memory_Usage_Server_{e}"]) / float(df2.at[last_index + 1,f"Total_GPU_Memory_Server_{e}"])) + float((df2.at[last_index + 1,f"CPU_Cores_Usage_Server_{e}"]) / float(df2.at[last_index + 1,f"Total_CPU_Cores_Server_{e}"]))

                else:
                    print(f"{sorted_tasks} is not a valid task")
                    break

                
            
            sat_service_req_count_per_episode = get_sat_service_req_count(tasks,selected_edge_servers,last_index + 1)
            
            #step reward
            step_reward_array.append(get_step_reward(selected_edge_servers,i))

            avg_wt = 0
            for e in selected_edge_servers:
                avg_wt += get_waiting_time(selected_edge_servers,last_index + 1)/total_steps

            print(f"Step {i} completed")

            dropping_cols = [col for col in df2.columns if 'Unnamed:' in col]
            #print("dropping cols: ", dropping_cols)

            df2 = df2.drop(columns=dropping_cols, axis=1)
            #print("cols dropped")
        
            df2.to_csv('/home/user/khushi/SafeTail 2.0/datasets/synthetic_edge_data.csv')

            print("---------------------------------------------------------------")
        #episodic reward    
        print()
        
        episodic_reward = get_episodic_reward(discount_factor, total_steps,selected_edge_servers,selected_tasks,tasks,sat_service_req_count_per_episode,avg_wt)
        
        #print (f"Episode {ep} completed")
        #print()
        #
        #print()

        episode_reward_array.append(episodic_reward)
        
        print(f"Episode {ep} completed")
        print("-----------------------------------------------------------------------------")
        print("-----------------------------------------------------------------------------")
        #plotting step reward
        plt.xlabel("Timestep")
        plt.ylabel("Step Reward")
        plt.title("Step Reward over timesteps")

        plt.plot(step_array,step_reward_array,label="Step Reward", color="green")
        plt.show()
        new_directory_step = ('/home/user/khushi/SafeTail 2.0/graphs/for_step_reward/'+ str(datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
        plt.savefig(new_directory_step + ".png")
        plt.close()


    

    #print(df2.head())
    #print(df2.tail())
    plt.xlabel("Episode")
    plt.ylabel("Episodic Reward")
    plt.title("Episodic Reward over episodes")
    #plt.plot(step_array,step_reward_array,label="Step Reward", color="green")
    #plt.plot(episode_array,episode_reward_array,label="Episodic Reward", color="red")
    new_directory_episode = ('/home/user/khushi/SafeTail 2.0/graphs/for_episode_reward/'+ str(datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
    plt.savefig(new_directory_episode + ".png")
    plt.close()

    #print("step array: ", step_array )
    print("-----------------------------------------------------------------------------")
    #print("episode_reward_array: ", episode_reward_array )
    print("-----------------------------------------------------------------------------")
    #print("step_reward_array: ", step_reward_array )
    print("-----------------------------------------------------------------------------")


    
    #plt.show()
    
    
controller()