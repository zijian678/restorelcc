import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pyvene as pv
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def find_most_similar_sbert(model, sentences, golden_sentence):
    golden_index = sentences.index(golden_sentence) if golden_sentence in sentences else -1

    # Encode all sentences
    embeddings = model.encode(sentences, convert_to_tensor=True)

    # Get similarity scores
    cosine_scores = util.pytorch_cos_sim(embeddings[golden_index], embeddings)[0]

    # Find the most similar (excluding the golden sentence itself)
    cosine_scores[golden_index] = -1  # Exclude self
    most_similar_index = torch.argmax(cosine_scores).item()

    simi_sentence = sentences[most_similar_index]

    # return {
    #     "golden_index": golden_index,
    #     "most_similar_index": most_similar_index,
    #     "similarity_scores": cosine_scores.tolist()
    # }
    return simi_sentence

def construct_boolq(samples):
    pos_samples = []
    neg_samples = []
    for i in samples:

        true_lab = i['label']
        text = i['text']
        if true_lab == ' yes':
            false_lab = ' no'
        else:
            false_lab = ' yes'
        current_sample1 = {
            'text':text+true_lab,
            'label':1
        }
        current_sample2 = {
            'text': text + false_lab,
            'label': 0
        }
        # print('current_sample1:',current_sample1)
        # print('current_sample2:',current_sample2)
        pos_samples.append(current_sample1)
        neg_samples.append(current_sample2)
    return pos_samples,neg_samples


def construct_rte(samples):
    pos_samples = []
    neg_samples = []

    for i in samples:

        true_lab = i['label']
        text = i['text']
        if true_lab == ' True':
            false_lab = ' False'
        else:
            false_lab = ' True'
        current_sample1 = {
            'text':text+true_lab,
            'label':1
        }
        current_sample2 = {
            'text': text + false_lab,
            'label': 0
        }
        # print('current_sample1:',current_sample1)
        # print('current_sample2:',current_sample2)
        pos_samples.append(current_sample1)
        neg_samples.append(current_sample2)
    return pos_samples,neg_samples

def construct_arc(samples):
    pos_samples = []
    neg_samples = []
    sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    for i in samples:

        true_lab = i['label']
        text = i['text']
        choices = i['choices']
        choices = [' '+kkk.strip() for kkk in choices]
        false_lab = find_most_similar_sbert(sentence_encoder, choices, true_lab)

        current_sample1 = {
            'text':text+true_lab,
            'label':1
        }
        current_sample2 = {
            'text': text + false_lab,
            'label': 0
        }
        # print('current_sample1:',current_sample1)
        # print('current_sample2:',current_sample2)
        pos_samples.append(current_sample1)
        neg_samples.append(current_sample2)
    return pos_samples,neg_samples

def construct_alpaca(samples):

    pos_samples = []
    neg_samples = []
    sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')

    all_ans = []
    for jj in samples:
        all_ans.append(jj['label'])
    embeddings = sentence_encoder.encode(all_ans)
    cosine_sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(cosine_sim_matrix, -1)
    most_similar_indices = np.argmax(cosine_sim_matrix, axis=1)
    most_similar_sentences = [all_ans[i] for i in most_similar_indices]
    # print('original sentence:',all_ans[0])
    # print('most_similar_sentences:',most_similar_sentences[0])

    for i,i_neg in zip(samples,most_similar_sentences):

        true_lab = i['label']
        text = i['text']

        # if i['input']:
        #     text = f"### Instruction:\n{i['instruction']}\n\n### Input:\n{i['input']}n\n### Response:"
        # else:
        #     text = f"### Instruction:\n{i['instruction']}n\n### Response:"

        # choices = i['choices']
        # choices = [' '+kkk.strip() for kkk in choices]
        false_lab = i_neg

        current_sample1 = {
            'text':text+true_lab,
            'label':1
        }
        current_sample2 = {
            'text': text + false_lab,
            'label': 0
        }
        # print('current_sample1:',current_sample1)
        # print('current_sample2:',current_sample2)
        pos_samples.append(current_sample1)
        neg_samples.append(current_sample2)
    return pos_samples,neg_samples

def wrapper(intervener):
    def wrapped(*args, **kwargs):
        return intervener(*args, **kwargs)
    return wrapped

class Collector():
    collect_state = True
    collect_action = False
    def __init__(self, multiplier, head):
        self.head = head
        self.states = []
        self.actions = []
    def reset(self):
        self.states = []
        self.actions = []
    def __call__(self, b, s):
        if self.head == -1:
            # print('b:', b.shape)
            # self.states.append(b[0, -1].detach().clone())  # original b is (batch_size, seq_len, #key_value_heads x D_head)
            self.states.append(b.detach().clone())  # original b is (batch_size, seq_len, #key_value_heads x D_head)
            # print('b2:', b.shape)
        else:
            # print('b:',b.shape)
            self.states.append(b[0, -1].reshape(32, -1)[self.head].detach().clone())  # original b is (batch_size, seq_len, #key_value_heads x D_head)
            # print('b:', b.shape)
        return b

def get_llama_activations_pyvene(collected_model, collectors, prompt, device):
    with torch.no_grad():
        prompt = prompt.to(device)
        output = collected_model({"input_ids": prompt, "output_hidden_states": True})[1]
    hidden_states = output.hidden_states
    hidden_states = torch.stack(hidden_states, dim = 0).squeeze()
    hidden_states = hidden_states.detach().cpu()#.numpy()
    head_wise_hidden_states = []
    for collector in collectors:
        if collector.collect_state:
            states_per_gen = torch.stack(collector.states, axis=0).cpu().numpy()
            head_wise_hidden_states.append(states_per_gen)
        else:
            head_wise_hidden_states.append(None)
        collector.reset()
    mlp_wise_hidden_states = []
    head_wise_hidden_states = torch.stack([torch.tensor(h) for h in head_wise_hidden_states], dim=0).squeeze()#.numpy()
    return hidden_states, head_wise_hidden_states, mlp_wise_hidden_states

def obtain_act(model, tokenizer, samples):
    layer_num = model.config.num_hidden_layers
    head_num = model.config.num_attention_heads
    head_dim = model.config.head_dim
    collectors = []
    pv_config = []
    for layer in range(model.config.num_hidden_layers):
        collector = Collector(multiplier=0,
                              head=-1)  # head=-1 to collect all head activations, multiplier doens't matter
        collectors.append(collector)
        pv_config.append({
            "component": f"model.layers[{layer}].self_attn.o_proj.input",
            "intervention": wrapper(collector),
        })
    collected_model = pv.IntervenableModel(pv_config, model)
    sample_acts = []
    for s in tqdm(samples):
        sen_ids = tokenizer(
            s['text'],
            return_tensors='pt',
            # padding='max_length',  # Pads sequences to max_length
            truncation=True,  # Truncates sequences longer than max_length
            max_length=512  # Sets fixed sequence length
        ).input_ids

        layer_wise_activations, head_wise_activations, _ = get_llama_activations_pyvene(collected_model, collectors, sen_ids,
                                                                                    device="cuda")
        # layer_wise_activations: (33, 145, 4096) head_wise_activations: (32, 145, 4096)
        # print('sen_ids:',sen_ids.shape,tokenizer.convert_ids_to_tokens(sen_ids[0]))
        # print('layer_wise_activations:',layer_wise_activations.shape,'head_wise_activations:',head_wise_activations.shape)
        current_act = head_wise_activations[:,-1,:]
        current_act = current_act.reshape(layer_num,head_num,head_dim)
        current_act = current_act.reshape(layer_num * head_num,head_dim) # torch.Size([1024, 128])
        # print('current_act:',current_act.shape)
        sample_acts.append(current_act)
    sample_acts = torch.stack(sample_acts)
    return sample_acts





def obtain_main_components(X,k = 10):
    X = X.float()
    U, S, Vt = torch.linalg.svd(X, full_matrices=False)
    # print('U:',U.shape,U,'S:',S.shape,S,'Vt:',Vt.shape,Vt)
    # ccc = U[0] * S
    # a0 = ccc.unsqueeze(0) @ Vt
    # print('X[0]:',X[0])
    # print('a0:',a0) X[0] = a0
    # U: torch.Size([512, 128]) S: torch.Size([128]) Vt: torch.Size([128, 128])
    # print('Vt:',Vt.shape) # Vt: torch.Size([128, 128])
    base_num = U.shape[1]
    added_act = []
    for i in range(base_num):
        coeff = U[:, i]
        # print('coeff:',coeff)
        # num_positive = (coeff > 0).sum().item()
        # num_negative = (coeff < 0).sum().item()
        # max_count = max(num_positive, num_negative)
        # ratio = max_count / (num_positive + num_negative)
        # if ratio > threshold:
        weight = torch.mean(coeff)
        current_act = weight * S[i] * Vt[i]
        added_act.append(current_act)


    V_padded = torch.zeros(Vt.shape[-1],Vt.shape[-1] , device=Vt.device)
    V_padded[:Vt.size(0), :] = Vt
    Vt = torch.nn.functional.normalize(V_padded,dim = -1, p = 2)
    # print('Vt:',Vt.shape,Vt)
    if not added_act:
        # added_act
        return None,Vt.half()
    else:
        added_act = torch.stack(added_act) # n*128
        # print('added_act:', added_act.shape)
        # added_act = added_act[]
        added_act = torch.sum(added_act[:k,:], dim=0) #k*128 - 128
        # print('added_act final:',added_act.shape)
        return added_act.half(),Vt.half()

def train_probing_classifier(pos_repre,neg_repre):
    # X = torch.cat([pos_repre, neg_repre], dim=0)
    # y = torch.cat([torch.ones(pos_repre.size(0)), torch.zeros(neg_repre.size(0))])

    # Train/validation split (preserving pairing)
    indices = list(range(len(pos_repre)))
    train_idx, val_idx = train_test_split(indices, test_size=0.3, random_state=42)

    # Create train and val datasets, keeping pairings aligned
    train_X = torch.cat([pos_repre[train_idx], neg_repre[train_idx]], dim=0)
    train_y = torch.cat([torch.ones(len(train_idx)), torch.zeros(len(train_idx))])

    val_X = torch.cat([pos_repre[val_idx], neg_repre[val_idx]], dim=0)
    val_y = torch.cat([torch.ones(len(val_idx)), torch.zeros(len(val_idx))])

    # train_X = torch.cat([x_added,train_X],dim = -1)
    # val_X = torch.cat([x_added, val_X], dim=-1)
    # print('train_X:',train_X.shape,train_X)
    # print('val_X:',val_X.shape,val_X)

    # Define model
    class LinearClassifier(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.linear = nn.Linear(input_dim, 1)

        def forward(self, x):
            return torch.sigmoid(self.linear(x))

    model = LinearClassifier(input_dim=train_X.shape[-1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    epochs = 200
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_X).squeeze()
        loss = criterion(outputs, train_y)
        loss.backward()
        optimizer.step()

        # print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(val_X).squeeze()
        preds = (val_outputs >= 0.5).float()
        accuracy = (preds == val_y).float().mean().item()

    # print(f"\nValidation Accuracy: {accuracy * 100:.2f}%")
    return accuracy

function_map  = {
    'boolq':construct_boolq,
    'arc_easy':construct_arc,
    'arc_challenge':construct_arc,
    'rte':construct_rte,
    'alpaca':construct_alpaca,
}

def obtain_main_vecs(data_name,data_num,original_model_path=None,pruned_model_path=None,comp_path = '',score_path = ''):

    # model_name = "baffo32/decapoda-research-llama-7B-hf"
    data_path = f'./processed_data/{data_name}_{str(data_num)}/train.pt'
    samples = torch.load(data_path)
    print('# of probing samples:',len(samples),samples[:1])
    contruct_func = function_map[data_name]
    pos_samples,neg_samples = contruct_func(samples)
    print('pos_samples:',len(pos_samples),pos_samples[:1])
    print('neg_samples:',len(neg_samples),neg_samples[:1])

    if data_name == "alpaca":
        samples = samples[:1000]
        pos_samples = pos_samples[:1000]
        neg_samples = neg_samples[:1000]


    #step 1 obtain final representations of positive and negatvie samples
    model = AutoModelForCausalLM.from_pretrained(
            original_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(original_model_path, use_fast=False)
    original_acts = obtain_act(model, tokenizer, samples) # original_acts: torch.Size([128, 1024, 128])
    # print('original_acts:',original_acts.shape) # original_acts: torch.Size([128, 1024, 128])
    pos_acts = obtain_act(model, tokenizer, pos_samples)
    neg_acts = obtain_act(model, tokenizer, neg_samples)
    del model
    #
    # step 2 obtain representations in pruned model
    model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path = pruned_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
    model.eval()
    pruned_acts = obtain_act(model, tokenizer, samples) # original_acts: torch.Size([128, 1024, 128])

    head_scores = []
    all_comps = []
    all_directions = []
    for idx in tqdm(range(pos_acts.shape[1])):
        # print('dealing with attention head num ',idx)
        ori_v = original_acts[:,idx,:]
        pruned_v = pruned_acts[:,idx,:]
        # print('ori_v:',ori_v.shape,'pruned_v:',pruned_v.shape)
        main_comp, v_norms = obtain_main_components(ori_v - pruned_v)
        all_directions.append(v_norms)
        # print('main_comp:',main_comp)
        pos_repre = pos_acts[:,idx,:]
        neg_repre = neg_acts[:,idx,:] # neg_repre: torch.Size([128, 128])
        # print('pos_repre:',pos_repre.shape,pos_repre)
        # print('neg_repre:', neg_repre.shape, neg_repre)
        if main_comp == None:
            score = 0
            all_comps.append(torch.zeros(neg_repre.shape[-1]))
        else:
            # pos_repre = pos_repre
            # neg_repre = neg_repre
            # print('main_comp:',main_comp)
            x_added = pruned_v + main_comp
            # print('x_added:',x_added.shape)
            pos_repre = torch.cat([x_added,pos_repre],dim = -1)
            neg_repre = torch.cat([x_added, neg_repre], dim=-1)
            # print('pos_repre:', pos_repre.shape,'neg_repre:', neg_repre.shape)
            # pos_repre: torch.Size([128, 256]) neg_repre: torch.Size([128, 256])
            score = train_probing_classifier(pos_repre.float(), neg_repre.float())
            all_comps.append(main_comp)
        head_scores.append(score)

    head_scores = torch.tensor(head_scores)
    # torch.save(head_scores,'head_scores_arc_easy.pt')
    # print('all_comps:',all_comps.shape)
    all_directions = torch.stack(all_directions)
    # print('all_directions:',all_directions.shape,all_directions)
    torch.save(all_directions,comp_path)
    print('saved components:',comp_path)
    # torch.save(all_comps, comp_path)


    kv,kinds = torch.topk(head_scores, k = len(head_scores))
    # print('kv:',kv)
    top_heads = []
    layer_num = model.config.num_hidden_layers
    for i in kinds:
        lid = int(i/layer_num)
        aid = int(i%layer_num)
        top_heads.append([lid,aid])
    top_heads = np.array(top_heads)


    np.save(score_path,top_heads)
    print('saved topk heads:',top_heads)


    return all_directions, top_heads # all_comps, top_heads

