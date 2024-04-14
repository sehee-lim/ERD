import yaml
import json
import os
import csv
import openai
import argparse
import traceback
import datetime
import pandas as pd
from tqdm import tqdm

import prompts
from logger import Logger

prompt_dict = prompts.prompt_dict

class Agent:
    def __init__(self, temperature: float, model: str) -> None:
        self.temperature = temperature
        self.model = model
        self.conversation = []   # list to save conversation history
    
    # use OpenAI API to get a response
    def respond(self) -> str:
        response = openai.ChatCompletion.create(
            model = self.model,
            messages = self.conversation,
            temperature = self.temperature
        )
        response = response.choices[0].message["content"]
        self.conversation.append({"role": "assistant", "content": response})   # add response to conversation history
        return response
      
      
      
      
class Debater(Agent):
    
    def __init__(self, temperature: float, model: str, prompt: str):
        super().__init__(temperature, model)
        logger.debug(prompt_dict[prompt])
        self.system_content = prompt_dict[prompt]["system"]
        self.user_content = prompt_dict[prompt]["user"]
        self.conversation.append({"role": "system", "content": self.system_content})
    
    
    def add_instruction(self, action: str, inps = None):
        # if inps is None:
        #     self.conversation.append({"role": "user", "content": self.user_content[action]})
        # else:
        #     self.conversation.append({"role": "user", "content": self.user_content[action].format(**inps)})
        
        if action == "initial_claim":
            question = self.user_content["initial_claim"]
            self.conversation.append({"role": "user", "content": question})
            
        elif action == "defense":
            question = self.user_content["defense"].format(counter_argument = refute)
            self.conversation.append({"role": "user", "content": question})
            
        elif action == "refute":
            question = self.user_content["refute"].format(user_speech = user_speech, dot = dot, counter_argument = initial_claim)
            self.conversation.append({"role": "user", "content": question})

        elif action == "dot":
            question = self.user_content["dot"].format(user_speech = user_speech)
            self.conversation.append({"role": "user", "content": question})
            
        elif action == "second_refute":
            question = self.user_content["second_refute"].format(counter_argument = defense)
            self.conversation.append({"role": "user", "content": question})
    
            
        return question, self.conversation


        
        

class Judge(Agent):
    def __init__(self, temperature: float, model: str, prompt: str):
        super().__init__(temperature, model)
        
        prompt_dict = prompts.prompt_dict
        self.system_content = prompt_dict[prompt]["system"]
        self.user_content = prompt_dict[prompt]["user"]
        self.conversation.append({"role": "system", "content": self.system_content})
        
        
    def summary_and_answering(self, entire_debate):
        self.system_content = self.system_content.format(user_speech = user_speech)
        self.conversation.append({"role": "user", "content": self.user_content["summary"].format(debate = entire_debate)})
    
        summary = self.respond()
        self.conversation.append({"role": "user", "content": self.user_content["answering"].format(debate = summary)})
        
        final_answering = self.respond()
        return summary, final_answering
        
        
    def answering(self, entire_debate):
        self.system_content = self.system_content.format(user_speech = user_speech)
        self.conversation.append({"role": "user", "content": self.user_content["answering"].format(debate = entire_debate)})
        
        final_answering = self.respond()
        return final_answering
    
    def extraction(self, entire_debate):
        self.conversation.append({"role": "user", "content": self.user_content["extraction"].format(user_speech = user_speech)})
        extraction = self.respond()
        self.conversation.append({"role": "user", "content": self.user_content["summary"].format(debate = entire_debate)})
        summary = self.respond()
        self.conversation.append({"role": "user", "content": self.user_content["answering"]})

        final_answering = self.respond()
        return extraction, summary, final_answering
        
        
        
            
        
            

# argument parser setup   
parser = argparse.ArgumentParser()

parser.add_argument('--openai_api_key', type = str, required = True)
parser.add_argument('--rounds', type = int, required = False)   # number of debate rounds
parser.add_argument('--summary', action = 'store_true')   # whether the judge should summarize the debate

parser.add_argument('--model_debater1', type = str, required = True)
parser.add_argument('--model_debater2', type = str, required = True)
parser.add_argument('--model_judge', type = str, required = True)

parser.add_argument('--prompt_cfg_debater1', type = str, required = True) 
parser.add_argument('--prompt_cfg_debater2', type = str, required = True)
parser.add_argument('--prompt_cfg_judge', type = str, required = True)

parser.add_argument('--subset', action = 'store_true')   # flag to indicate use of a subset
parser.add_argument('--subset_idx', default = 1, type = str, required = False)   # index of the subset

parser.add_argument('--rerun', action = 'store_true')   # flag to indicate a rerun of the process
parser.add_argument('--folder_stopped', type = str, required = False)   # folder name where the process was previously stopped



if __name__ == "__main__":
    args = parser.parse_args()
    openai.api_key = args.openai_api_key

    
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
        dataset = pd.read_csv(f"../data/subset-{args.subset_idx}.csv", encoding = 'utf-8').reset_index()
    else:
        # load the full dataset
        dataset = pd.read_csv("../data/Annotated_data.csv", encoding = 'utf-8').reset_index()
        dataset["User speech"] = dataset["Patient Question"]
    
    logger.debug(dataset.columns)
    
    
    
    # csv file to store results
    result = []
    columns = ['Distortion Assessment', 'Distortion Type', 'Id_Number', 'Dominant Distortion', 'Secondary Distortion', 'User_Speech']
    csv_file_path = f"{output_dir}/{run_name}-exp_result.csv"
    
    # create a new file if the file does not exist
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, 'w', newline = '', encoding = 'utf-8') as file:
            writer = csv.DictWriter(file, fieldnames = columns)
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
            # if i > 5:
            #     break
        
            debater1 = Debater(temperature = 0.1, model = args.model_debater1, prompt = args.prompt_cfg_debater1)
            debater2 = Debater(temperature = 0.1, model = args.model_debater2, prompt = args.prompt_cfg_debater2)
            judge = Judge(temperature = 0.1, model = args.model_judge, prompt = args.prompt_cfg_judge)  
            
            sample = dataset.iloc[i]
            user_speech = sample["User speech"]
            
            debate = {"User speech": user_speech}
            # initial claim by debater1
            q1, dot_prompt = debater1.add_instruction(action = "dot")   # dot thought process
            debate["q1"] = q1
            dot = debater1.respond()
            debate["dot"] = dot
            q2, initial_claim_prompt = debater1.add_instruction(action = "initial_claim")
            debate["q2"] = q2
            initial_claim = debater1.respond()
            
            
            debate["Initial_claim"] = initial_claim
            
            # refutation by debater2
            q3, refute_prompt = debater2.add_instruction(action = "refute")
            debate["q3"] = q3
            refute = debater2.respond()
            debate["Refute"] = refute
                
            # defense by debater1
            q4, defense_prompt = debater1.add_instruction(action = "defense")
            debate["q4"] = q4
            defense = debater1.respond()   
            debate["Defense"] = defense
            
            # second refutation by debater2
            q5, second_refute_prompt = debater2.add_instruction(action = "second_refute")
            debate["q5"] = q5
            second_refute = debater2.respond()
            debate["Second Refute"] = second_refute
            
            for_judge = {"debater1_first_claim": dot + initial_claim,
                         "debater2_refute": refute,
                         "debater1_defense": defense,
                         "debater2_second_refute": second_refute}
            if args.summary:
                summary, final_answer = judge.summary_and_answering(entire_debate = for_judge)
                prediction = json.loads(final_answer)
                debate["Summary"] = summary
            else:
                final_answer = judge.answering(entire_debate = for_judge)
                # final_answer = judge.answering(entire_debate = for_judge)
                prediction = json.loads(final_answer)
                # debate["Extraction"] = extraction
                # debate["Summary"] = summary
                
            
            user_contents = [item['content'] for item in judge.conversation if item['role'] == 'user']
            debate["Judge questions"] = user_contents
            debate["Judge"] = final_answer
            
            
            # logger.debug(debate)
            
            log = dict(prediction, **{"Id_Number": sample['Id_Number'],
                                    "Dominant Distortion": sample['Dominant Distortion'], 
                                    "Secondary Distortion": sample["Secondary Distortion (Optional)"],
                                    "User_Speech": user_speech})
                
            
            with open(f"{output_dir}/{run_name}-sample{i}.txt", "a", encoding = 'utf-8') as f:
                for k, v in debate.items():
                    f.write(f"\n[{k}]\n{v}")
                
            # append the log entry as a new row to the CSV file
            with open(csv_file_path, "a", newline = "", encoding = "utf-8") as file:
                writer = csv.DictWriter(file, fieldnames = columns)
                writer.writerow(log)
                
        # user-initiated program interruption
        except KeyboardInterrupt:
            print("Program interrupted by the user.")
    
            # save the current index for resuming later
            with open(f'{output_dir}/last_index.json', 'w') as f:
                json.dump({'last_index': i}, f)
                    
            break   # exit the loop
        
        # any unexpected exceptions during program execution
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
            