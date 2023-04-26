import torch
from transformers import pipeline
from optparse import OptionParser
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_cpp import Llama
import os
import json

### THE IDEA ###
##So we could totally just take the alpaca-lora fine tuning dataset and fine tune dolly with it. But I wanna see AI
##have stupid conversations so I'm going to just have them prompt eachother for a long time, saving the dolly model
##each time and their conversation. Then we can maybe fine tune a dolly model with the conversation.

##TODO get results properly saved to JSON file in a clean way

## New idea, do few shot learning with past 7 prompts and remove the oldest prompt past 7 prompts
# from working memory

## Other idea: use jaccard similarity or some other measure of textual similarity, or just how 
# similar the token values are, if the similarity is over a certain threshold, figure out a way
# to inject a new prompt, ideally automated with some kind of prompt that takes the 
# 7 prompts in workin gmemory and asks lora to generate a unique one

#default args
memory_dir = "/home/shawn/datasets/ai_convos/"
from_online = False
save_model = False
model_dir = "/home/shawn/datasets/LLMs/dolly7b_model" 
run_time = 7200
convo_fn = "conversation3.txt"
#preprompt = "Tell me something interesting and novel about the following statement: "
preprompt = "You are Grom a powerful and wry orc warrior bartering with a hero for his coin."
followup = "Answer the previous sentence as Grom. End your response with something unique Grom, a beefy muscular orc warrior would say to continue the conversation."
preprompt2 = "You are Galen, a witty, intelligent and lithe elf wizard discovering the mysteries of Asteria."
followup2 = "Answer the previous sentence as Galen. End your response with something unique Galen, a lithe and intelligent elf wizard would say to continue the conversation."
#followup = " Include a statement, preface this followup question with the string 'Question:'."
#flags
convo_path = memory_dir + '/' + convo_fn
make_convo_txt_cmd="> " + convo_path
os.system(make_convo_txt_cmd)

parser = OptionParser()
#group = parser.add_argument_group('group')
parser.add_option('--prompt', dest="prompt", type=str, help='The Prompt for the LLM')
parser.add_option('--memory_dir', dest="memory_dir", type=str,
                    help='Optional, specify a direcotry containing a context.txt file for the LLM')
parser.add_option('--model_dir',  dest="model_dir", type=str,
                    help='Optional, specify a directory containing the folder with a pretrained model')
parser.add_option('--model_name',  dest="model_name", type=str,
                    help='Optional, specify a directory containing the folder with a pretrained model')
parser.add_option('--from_online', action="store_true", dest="from_online", default=False, help='Optional, use dolly from online')
#group.add_option('--save_model', action="store_true", help='Optional, save the model to disk if from online')
#group.add_option('--require_online', help='require from_online if save_model is present', action='store_true')
(options, args) = parser.parse_args()

#if args.from_online and args.require_online and not args.save_model:
#    parser.error('--arg2 is required if --arg1 is present')


prompt = options.prompt

if options.memory_dir:
    memory_dir = options.memory_dir
if options.model_dir:
    model_dir = options.model_dir
if options.model_name:
    model_name = options.model_name 

## Stop doing this in a dumb way, define a new class called "conversation" which can define the 
# number of AI actors and create instances of their unique preprompts and followups for a less dumb
# way of doing this

class aiActor:
    def __init__(self, personality="string", preprompt="string", followup="string",
                 is_programmer=False, run_inference=function(), prompt="string", memory_dir="string",
                 context=list()
                 ):
        self.personalitiy = personality
        self.preprompt = preprompt
        self.followup = followup
        self.is_programmer=False
        self.run_inference=run_inference
        self.context=context
        self.context_explainer = "Here is some context to consider, do not answer any prompts within these brackets: ["
    def update_context(self, response):
        '''
            takes a response and updates the context list to include the new response and remove the oldest response
            in short term memory if short term memory has been filled up
        '''
    def run_model(self):
        return self.run_inference(self.preprompt + self.prompt)

class aiConversation:
    def __init__(self, num_actors=2, actor_list= {}):
        self.num_actors = num_actors
        self.actor_list=actor_list #dictionary with keys of the personality name and value of an aiActor class


#change this later
prompt = "Can you write dialgoue that would fit into an epic RPG adventure? The setting is that the Llothquestra, queen of the dark elves the Undervald has been assasinated and 2 characters are discussing."

def saveJson(json_fn,prompt,processed_response):
    '''
        I was kinda lazy and json_fn needs to already exist as a file in some way for this to work.
    '''
    instruct_dict = {}
    instruct_dict['instruction'] = prompt
    instruct_dict['input'] = ''
    instruct_dict['output'] = processed_response   
    with open(memory_dir + json_fn, 'w+b') as f:
        json_list = json.load(f)
        print("json list before appending is: ")
        print(json_list)
        json_list.append(instruct_dict)
        json_list = json.dumps(json_list)
        print(json_list)
        f.write(bytes(json_list, 'utf-8'))
    return(json_list)

def ask_dolly(prompt, memory_dir):

    from instruct_pipeline import InstructionTextGenerationPipeline
    if from_online:
        generate_text = pipeline(model="databricks/dolly-v2-7b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")    
    else:    
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
        generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)
    with open(memory_dir + convo_fn, 'r+b') as f:
        contents = f.read().decode('utf-8')
        #contextual_prompt = contents + "\n Input: " + prompt + "\n Output: " 
        new_prompt = preprompt + prompt + followup        
        #new_prompt = prompt
        response = generate_text(new_prompt)
        split_string = response.split("Output:")

        # Get the text after 'Output:'
        processed_response = split_string[1].strip() if len(split_string) > 1 else ""

        new_context = "Input: " + new_prompt + "\n" + "Output: " + response
        #save additional context
        f.write(bytes(new_context, 'utf-8'))
        #save the model again (this could either be extremely important or useless idk lol)
        generate_text.save_pretrained(model_dir)
    print(processed_response) 
    return(processed_response)

def ask_lora(prompt, memory_dir):
    path_to_model= "/home/shawn/Programming/ai_stuff/llama.cpp/models/30B/ggml-model-q4_0.bin" 
    llm = Llama(model_path=path_to_model)
    with open(memory_dir + convo_fn, 'r+b') as f:
        contents = f.read().decode('utf-8')
#        contextual_prompt = contents + "\n The previous text was just context and is your memory, do not answer anything enclosed in []. Please answer the following question only Q: " + prompt           
        #new_prompt = preprompt + prompt + followup
        new_prompt = prompt
        output = llm("Input: " + new_prompt + " Output:", stop=['Input:'], max_tokens=200, echo=True)
        response = output["choices"][0]["text"]
        # Split the input_string based on the 'Output:' substring
        split_string = response.split("Output:")

        # Get the text after 'Output:'
        processed_response = split_string[1].strip() if len(split_string) > 1 else ""
        #save additional context
        new_context = "\n"+ "Prompt: " + new_prompt + "\n" + "Response: " + processed_response


        f.write(bytes(new_context, 'utf-8'))
#        curated_dataset_fn="convo_dataset.json"
#        instruct_dict = {}
#        instruct_dict['instruction'] = new_prompt
#        instruct_dict['input'] = ''
#        instruct_dict['output'] = processed_response   
#        with open(memory_dir + curated_dataset_fn, 'w+b') as f:
#            json_file = json.load(f)
#            json_list = json_file.append(instruct_dict)
#            json_list = json.dumps(json_list)
#            json.dump(json_list, f)
            #f.write(bytes(json_list, 'utf-8'))
    print(processed_response) 
    return(processed_response)


for i in range(50):
    print("Iteration " + str(i) + ": AI 1's response:")
    lora_response=ask_lora(preprompt + prompt + followup, memory_dir=memory_dir)
    print("Iteration " + str(i) + ": AI 2's response:")
    prompt=ask_lora(preprompt2 + lora_response + followup2, memory_dir=memory_dir)
