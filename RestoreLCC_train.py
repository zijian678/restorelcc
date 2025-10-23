import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import json, os

from transformers import AutoTokenizer, TrainingArguments, DataCollatorWithPadding, AutoModelForCausalLM, \
    get_linear_schedule_with_warmup, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig, AdamW, set_seed, \
    logging
from models.modeling_llama import LlamaModel, LlamaForCausalLM
import torch
import torch.nn as nn
from trl import DataCollatorForCompletionOnlyLM
import argparse
import numpy as np
import wandb
import random
from utils.dataloaders import  obtain_sepc_datasets
from utils.trainers import  CustomSFTTrainer
from utils.components import obtain_main_vecs



parser = argparse.ArgumentParser()

#check the following arguments
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--num_epoch', type=int, default=3)
parser.add_argument('--use_topk_heads', type=int, default=256,
                    help='The number of top attention heads to select')
parser.add_argument('--spec_task', type=str, default='alpaca')
parser.add_argument('--num_train_samples', type=int, default=2000)
parser.add_argument('--pruned_model_path', type=str, default="wanda_pruned_llama_7B_50")
parser.add_argument('--base_model_name', type=str, default='llama_7B', help='The model base to train on')


parser.add_argument('--train_batch', type=int, default=8)
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--output_dir', type=str,
                    default="./finetuned_checkpoints")
parser.add_argument('--eval_batch', type=int, default=8)
parser.add_argument('--run_mode', type=str, default='train',
                    help='The mode to run the script: train or train_wandb. Train: train the model; train_wandb: train the model and log the results to wandb.')
parser.add_argument('--applied_module', type=str, default='attention',
                    help='The modules to apply lofit; attention by default')
parser.add_argument('--applied_layers', type=str, default=None,
                    help='The list of layers to apply lofit; None by default and it means apply lofit to all layers')
parser.add_argument('--l1_lambda', type=float, default=0, help='l1 regularization lambda for lofit', required=False)

parser.add_argument('--lofit_component', type=str, default='v',
                    help='Choose the components to apply acfit. A: head selection step; v: bias tuning step',
                    required=False)
parser.add_argument('--ft_method', type=str, default='lofit', help='fine-tuning method to apply')

parser.add_argument('--lofit_heads', type=str,
                    default="arc_easy_top_head.npy",
                    help='Load a .npy file where the top heads from the head selection step are stored', required=False)

parser.add_argument('--hf_cache_dir', type=str, default='../cache',
                    required=False, help='The cache directory for huggingface models')
parser.add_argument('--device', type=str, default='cuda', required=False,
                    help='The device to load the model; cuda by default')
parser.add_argument('--save_strategy', type=str, default='no', required=False,
                    help='The strategy to save the model: best: only save the best model; no: do not save the model')


args = parser.parse_args()
### Turn Wandb log on if it is in train mode
if args.run_mode == 'train_wandb':
    wandb.init(mode="online", name=args.output_dir.split("/")[-1])
else:
    wandb.init(mode="disabled")
### Load training hyperparametres
lr = float(args.lr)
train_batch_size = int(args.train_batch)
eval_batch_size = int(args.eval_batch)

num_epoch = int(args.num_epoch)
applied_module = args.applied_module
l1_lambda = args.l1_lambda

output_dir = args.output_dir
device = args.device
lofit_heads = args.lofit_heads
topk_heads = args.use_topk_heads



if args.applied_layers is not None:
    applied_layers = list(map(int, args.applied_layers.split(',')))
else:
    applied_layers = None
## Set all random seeds for reproducibility
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
set_seed(seed)

logging.set_verbosity_error()
### Maps of model names and task names
### If you want to use your own model, please add the model name to the map
models_map = {
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf',
    'llama2_7B': 'meta-llama/Llama-2-7b-hf',
    'llama2_13B': 'meta-llama/Llama-2-13b-hf',
    'llama_7B':'baffo32/decapoda-research-llama-7B-hf',
    'llama_13B':'huggyllama/llama-13b',
    'llama_30B':'huggyllama/llama-30b',
    'llama3_8B':'meta-llama/Meta-Llama-3-8B',
    'vicuna_7b':"lmsys/vicuna-7b-v1.5",
    'vicuna_13b':"lmsys/vicuna-13b-v1.5",
    'Tulu_8b':"allenai/Llama-3.1-Tulu-3.1-8B",
    'tulu2_7b':'allenai/tulu-2-7b'
}


if not args.base_model_name in models_map:
    raise ValueError(f'The base model {args.base_model_name} is not supported')
### Load tokenizers and models
model_name = models_map[args.base_model_name]
cache_dir = args.hf_cache_dir
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir,use_fast=False)

### Use right padding for training
tokenizer.padding_side = 'right'
torch_dtype = torch.bfloat16
bf16 = True




comp_path = f'./processed_data2/main_comps/{args.base_model_name}_{args.spec_task}_{str(args.num_train_samples)}.pt'
head_score_path = f'./processed_data2/main_comps/{args.base_model_name}_{args.spec_task}_{str(args.num_train_samples)}.npy'
print('comp_path:',comp_path)
print('head_score_path:',head_score_path)
if os.path.exists(comp_path):
    main_comps = torch.load(comp_path)
else:
    main_comps,tohe =  obtain_main_vecs(args.spec_task, args.num_train_samples, original_model_path=model_name,
                                        pruned_model_path=args.pruned_model_path, comp_path=comp_path, score_path=head_score_path)
print('main_comps:',main_comps.shape,main_comps)
if lofit_heads is not None:
    assert '.npy' in lofit_heads
    ### Only use the topk_heads heads
    print(f'Number of Attention Heads Used For Training: {topk_heads}')
    lofit_heads = np.load(head_score_path)[:topk_heads, :]
    ### Convert np array to list of tuples
    lofit_heads = list(zip(lofit_heads[:, 0], lofit_heads[:, 1]))
print('lofit_heads:',lofit_heads)

if args.ft_method == 'lofit':
    if 'llama' or 'vicuna' or 'tulu' in model_name:

        model = LlamaForCausalLM.custom_from_pretrained(pretrained_model_name_or_path = args.pruned_model_path,
                                                        device_map=device,
                                                        cache_dir=cache_dir,
                                                        applied_module=applied_module,
                                                        applied_layers=applied_layers,
                                                        torch_dtype=torch_dtype)
    else:
        raise ValueError(f'Fine-tuning method {args.ft_method} for {model_name} is not supported!')
else:
    raise ValueError(f'Fine-tuning method {args.ft_method} is not supported!')
### Define padding
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(model.config.vocab_size + 1)

count = 0
if args.run_mode != 'test':
    ### First freeze all pretrained parameters
    for param in model.parameters():
        param.requires_grad = False
    trainable_params = []
    num_params = 0
    ### Unfreeze LoFiT parameters for training
    for i in range(model.config.num_hidden_layers):
        if applied_module == 'attention':
            # if args.lofit_component == 'A':
            #     attn_A = model.model.layers[i].self_attn.attn_A
            #     for j, module in enumerate(attn_A):
            #         trainable_params.append(module)
            #         module.requires_grad = True
            #         num_params += module.numel()
            if args.lofit_component == 'v':
                attn_v = model.model.layers[i].self_attn.attn_v
                for j, module in enumerate(attn_v):
                    if lofit_heads is None or (i, j) in lofit_heads:
                        trainable_params.append(module)
                        module.requires_grad = True
                        num_params += module.numel()
                        count += 1
                attn_b = model.model.layers[i].self_attn.attn_b
                for j, module in enumerate(attn_b):
                    if lofit_heads is None or (i, j) in lofit_heads:
                        trainable_params.append(module)
                        module.requires_grad = True
                        num_params += module.numel()
                        count += 1
        else:
            raise ValueError(f'Fine-tuning {applied_module} is supported yet!')
    print('trainable params:', num_params)
    # optimizer = AdamW(trainable_params, lr=lr)
if args.save_strategy == 'best':
    save_strategy = 'epoch'
    load_best_model_at_end = True
    save_total_limit = 1
elif args.save_strategy == 'no':
    save_strategy = 'no'
    load_best_model_at_end = False
    save_total_limit = None
else:
    raise ValueError(f'Save strategy {args.save_strategy} is not supported')
training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=lr,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=eval_batch_size,
    num_train_epochs=num_epoch,
    evaluation_strategy="epoch",
    save_strategy=save_strategy,
    load_best_model_at_end=load_best_model_at_end,
    save_total_limit=save_total_limit,
    report_to='wandb',
    logging_strategy='epoch',
    seed=seed,
    do_train=True,
    do_eval=True,
    bf16=bf16,
disable_tqdm=False,
)
torch.autograd.set_detect_anomaly(True)

datasets = {
'train': None,
'val':None
}

trainer = CustomSFTTrainer


train_data_path = f'./processed_data/{args.spec_task}_{str(args.num_train_samples)}/train.pt'
valid_data_path = f'./processed_data/{args.spec_task}_{str(args.num_train_samples)}/valid.pt'
print('train_data_path:',train_data_path)
print('valid_data_path:',valid_data_path)
train_datasets = obtain_sepc_datasets(train_data_path ) # ,max_num=100
val_datasets = obtain_sepc_datasets(valid_data_path ) # ,max_num=100
datasets['train'] = train_datasets
datasets['val'] = val_datasets
print(f"Number of train samples: {len(datasets['train'])}")
print(f"Number of val samples: {len(datasets['val'])}")
## check
#

if args.spec_task == 'alpaca':
    response_template_with_context = "\n\n### Response:"
else:
    response_template_with_context = "\nAnswer:"
print('response_template_with_context:',response_template_with_context)

if 'llama' or 'vicuna' or 'tulu' in model_name:
    ### Special thing about llama tokenizer

    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]
elif 'gemma' in model_name:
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[1:]
### DataCollatorForCompletionOnlyLM is used for updating loss ONLY on the response
### If you want to do standard LM loss (i.e. loss update on both the prompt and the response), you don't need to use this data collator
data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

trainer = trainer(
    model,
    train_dataset=datasets['train'],
    eval_dataset=datasets['val'],
    dataset_text_field='text',
    tokenizer=tokenizer,
    max_seq_length=500,
    data_collator=data_collator,
    args=training_args,
    peft_config=None
)
layer_num = model.config.num_hidden_layers
head_num = model.config.num_attention_heads
head_dim = model.config.head_dim

def check(model,lofit_heads,path = None):
    for i in lofit_heads:
        lid = i[0]
        aid = i[1]
        print(lid,'***',aid)
        print(model.model.layers[lid].self_attn.attn_v[aid])
        print(model.model.layers[lid].self_attn.attn_b[aid])
        print(model.model.layers[lid].self_attn.main_comp[aid])

if args.run_mode != 'test':
    trainer.l1_lambda = l1_lambda
    if args.ft_method == 'lofit':
        for i in range(model.config.num_hidden_layers):
            if applied_module == 'attention':
                # if args.lofit_component == 'A':
                #     attn_A = model.model.layers[i].self_attn.attn_A
                #     for j, module in enumerate(attn_A):
                #         ### Use miu_{A} = 0, sigma_{A} = 1e-3 as the default
                #         nn.init.normal_(module, mean=0, std=1e-3)
                if args.lofit_component == 'v':
                    attn_v = model.model.layers[i].self_attn.attn_v
                    for j, module in enumerate(attn_v):
                        if lofit_heads is None or (i, j) in lofit_heads:
                            ### Use miu_{v} = 0, sigma_{v} = 1e-3 as the default
                            nn.init.normal_(module, mean=0, std=1e-3)
                            # module.data.copy_(torch.ones_like(module))
                    attn_b = model.model.layers[i].self_attn.attn_b
                    for j, module in enumerate(attn_b):
                        if lofit_heads is None or (i, j) in lofit_heads:
                            ### Use miu_{v} = 0, sigma_{v} = 1e-3 as the default
                            nn.init.normal_(module, mean=0, std=1e-3)
                            # module.data.copy_(torch.ones_like(module))
                    ## added initialization
                    comp = model.model.layers[i].self_attn.main_comp
                    for co, module in enumerate(comp):
                        if lofit_heads is None or (i, co) in lofit_heads:
                            ### Use miu_{v} = 0, sigma_{v} = 1e-3 as the default
                            # nn.init.normal_(module, mean=0, std=1e-3)
                            attn_idx = i *layer_num + co
                            # print('attn_idx:',attn_idx)
                            weight = main_comps[attn_idx].half().to('cuda')
                            # print('weight:',weight)
                            module.data.copy_(weight)
                            # print('module.data:',module.data.shape,module.data)
    # check(model, lofit_heads)
    trainer.train()

temp_dir = 'alpaca_finetuned_model1'
model.save_pretrained(temp_dir)
tokenizer.save_pretrained(temp_dir)


model.seqlen = 2048




# eval zero-shot
accelerate=False

if args.spec_task == 'alpaca':
    task_list = ["boolq", "rte","hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa"]
else:
    task_list = [args.spec_task]
num_shot = 0
model.eval()
# check(model, lofit_heads)
from eval_new import eval_zero_shot
full_model_name = models_map[args.base_model_name]
results = eval_zero_shot(full_model_name, temp_dir,model, tokenizer, task_list, num_shot, accelerate)
print("********************************")
print("zero_shot evaluation results")
print(results)
print('lr:',args.lr,'num_epoch:',args.num_epoch,'use_topk_heads:',args.use_topk_heads,'task:',args.spec_task,
      'num_train_samples:',args.num_train_samples)