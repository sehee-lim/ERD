import os
import csv
import json
import yaml
import openai
import argparse
import traceback
import datetime
import pandas as pd
from tqdm import tqdm

import prompts
from logger import Logger




# argument parser setup   
parser = argparse.ArgumentParser()

parser.add_argument('--openai_api_key', type = str, required = True)

parser.add_argument('--model', type = str, required = True, default = "gpt-3.5-turbo")
parser.add_argument('--prompt_cfg', type = str)

parser.add_argument('--subset', action = 'store_true')   # flag to indicate use of a subset
parser.add_argument('--subset_idx', default = 1, type = str, required = False)   # index of the subset

parser.add_argument('--rerun', action = 'store_true')   # flag to indicate a rerun of the process
parser.add_argument('--folder_stopped', type = str, required = False)   # folder name where the process was previously stopped

def get_response(messages):
    response = openai.ChatCompletion.create(
            model = args.model,
            messages = messages,
            temperature = 0.1
        )
    response = response.choices[0].message["content"]
    return response


if __name__ == '__main__':
    args = parser.parse_args()
    
    # enforce subset_idx argument when subset flag is enabled
    if args.subset and not args.subset_idx:
        parser.error("The --subset_idx argument is required when --subset is used.")
        
    # enforce folder_stopped argument when rerun flag is enabled
    if args.rerun and not args.folder_stopped:
        parser.error("--folder_stopped is required when --rerun is used.")
        
    # directory to save the output
    if args.rerun:
        output_dir = f'./output/{args.folder_stopped}/'
    else:
        output_dir = f'./output/{str(datetime.datetime.now().strftime("%y%m%d-%H%M%S"))}'
    
    # set the run_name of the logger
    if args.subset:
        run_name = f"SubsetIDX-{args.subset_idx}"
    else:
        run_name = "FullDataset"
        
    # initialize a logger
    logger = Logger(run_name, output_dir)
    logger.addFileHandler(f'{run_name}-log.txt')   # add a file handler for logging
    
    for k, v in vars(args).items():
        # skip logging the OpenAI API key
        if v == args.openai_api_key:
            continue
        logger.debug(f'\n\t{k}: {v}')
    
    
    
    if args.subset:
        # load a specific subset of the dataset
        dataset = pd.read_csv(f"data/subset_{args.subset_idx}.csv", encoding = 'utf-8').reset_index()
    else:
        # load the full dataset
        dataset = pd.read_csv("data/Annotated_data.csv", encoding = 'utf-8').reset_index()
        dataset["User speech"] = dataset["Patient Question"]
        
    logger.debug(dataset.columns)
    
    os.environ["MODEL"] = args.model
    openai.api_key = args.openai_api_key
    
    prompt_dict = prompts.prompt_dict   # retrieve the dictionary of prompts
    prompt = prompt_dict[args.prompt_cfg]
    logger.debug(prompt)   # log the specific prompt configuration
    
    # csv files to store results
    result = []
    columns = ['Distortion Assessment', 'Distortion Type', 'Id_Number', 'Dominant Distortion', 'Secondary Distortion', 'User_Speech']
    csv_file_path = f"{output_dir}/{run_name}-exp_result.csv"
    
    # if the file does not exist
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, 'w', newline = '', encoding = 'utf-8') as file:
            writer = csv.DictWriter(file, fieldnames = columns)   # create a new CSV file 
            writer.writeheader()
            
    if args.rerun:
        file_path = f'output/{args.folder_stopped}/{run_name}-exp_result.csv'
        result_df = pd.read_csv(file_path)
    else:
        result_df = pd.DataFrame(result, columns = columns)
        result_df.to_csv(f'{output_dir}/{run_name}-exp_result.csv', mode = "w", header = True, index = False)
        
        
    # load the last processed index
    try:
        with open(f'output/{args.folder_stopped}/last_index.json', 'r') as f:
            last_index = json.load(f)['last_index']
    except FileNotFoundError:
        last_index = 0
        
    
    
    for i in tqdm(range(last_index, len(dataset))):
        try:
            # if i > 1:
            #     break
        
            sample = dataset.iloc[i]
            user_speech = sample["Patient Question"]
            
            messages = [
                {"role": "system", "content": prompt["system"]}
            ]
            
            for step, prompt_content in prompt["user"].items():
                dict = {"role": "user", "content": prompt_content.format(user_speech = user_speech)}
                messages.append(dict)
                response = get_response(messages = messages)
                response_dict = {"role": "assistant", "content": response}
                messages.append(response_dict)
            logger.debug(messages)
            pred_yn = messages[-3]["content"]
            pred_type = messages[-1]["content"]
            log = {"Distortion Assessment": pred_yn,
                   "Distortion Type": pred_type,
                   "Id_Number": sample['Id_Number'],
                   "Dominant Distortion": sample['Dominant Distortion'], 
                   "Secondary Distortion": sample["Secondary Distortion (Optional)"],
                   "User_Speech": user_speech}
            
            with open(f"{output_dir}/{run_name}-sample{i}.txt", "a", encoding = 'utf-8') as f:
                for message in messages[1:]:
                    f.write(f"\n{message['content']}\n")
                    
            # append the log entry as a new row to the CSV file
            with open(csv_file_path, "a", newline = "", encoding = "utf-8") as file:
                writer = csv.DictWriter(file, fieldnames = columns)
                writer.writerow(log)
            
        except KeyboardInterrupt:
            print("\nProgram interrupted by the user.")
            
            # save the current index for resuming later
            with open(f'{output_dir}/last_index.json', 'w') as f:
                json.dump({'last_index': i}, f)
                    
            break   # exit the loop
        
        except Exception as e:
            error_message = f"Error occured at iteration-{i}: {e}"
            print(error_message)
            logger.error(error_message)   # log the error message
            
            traceback_info = traceback.format_exc()
            logger.error(traceback_info)   # log the complete error traceback
            
            # save the current index for resuming later
            with open(f"{output_dir}/last_index.json", "w") as f:
                json.dump({'last_index': i}, f)
            
            break   # exit the loop