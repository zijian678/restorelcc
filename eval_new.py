import torch
import torch.nn as nn
import fnmatch
def eval_zero_shot(model_name,model_dir, model, tokenizer, task_list=["boolq","rte","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"],
        num_fewshot=0,use_accelerate=False,device='auto', add_special_tokens=False):
    from lm_eval import tasks, evaluator
    def pattern_match(patterns, source_list):
        task_names = set()
        for pattern in patterns:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        return list(task_names)
    task_names = pattern_match(task_list, tasks.ALL_TASKS)
    model_args = f"pretrained={model_name},cache_dir=m{model_dir},dtype=bfloat16" # ,dtype=bfloat16
    limit = None
    # use_accelerate = True
    if "70b" in model_name or "65b" in model_name:
        limit = 2000
    if use_accelerate:
        model_args = f"pretrained={model_name},cache_dir=./llm_weights,use_accelerate=True"
    results = evaluator.simple_evaluate(
        model="hf-causal-experimental",
        model_args=model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=4,
        device='cuda',
        no_cache=True,
        limit=limit,
        description_dict={},
        decontamination_ngrams_path=None,
        check_integrity=False,
        pretrained_model=model,
        tokenizer=tokenizer,
        add_special_tokens=add_special_tokens,
    )

    return results