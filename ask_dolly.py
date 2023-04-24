import torch
from transformers import pipeline
from optparse import OptionParser
import importlib
import importlib.util
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

#default args
memory_dir = "/home/shawn/datasets/dolly_memory/"
from_online = False
save_model = False
model_dir = "/home/shawn/datasets/LLMs/dolly7b_model" 

#flags
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

from instruct_pipeline import InstructionTextGenerationPipeline
if from_online:
    generate_text = pipeline(model="databricks/dolly-v2-7b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")    
else:    
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
    generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)

def ask_dolly(prompt, memory_dir):
    with open(memory_dir + 'context.txt', 'r+b') as f:
        contents = f.read().decode('utf-8')
        #contextual_prompt = contents + "\n Input: " + prompt + "\n Output: "          
        response = generate_text(prompt)
        new_context = "Input: " + prompt + "\n" + "Output: " + response
        #save additional context
        f.write(bytes(new_context, 'utf-8'))
        #save the model again (this could either be extremely important or useless idk lol)
        generate_text.save_pretrained(model_dir)
    print(response) 
    return(response)

ask_dolly(prompt, memory_dir)