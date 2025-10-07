"""
Sample from a trained model
"""
import time
import re
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "B\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 50 # number of samples to draw
max_new_tokens = 1000 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    e_token = encode("E")[0]
    

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# set output filename based on dataset name
dataset_name = checkpoint['config']['dataset'] if (init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']) else 'output' 
output_path = os.path.join("output/generated_tracks", f"generated_{dataset_name}_sample1.csv") #
raw_output_path = os.path.join("output/generated_tracks", f"raw_sample_new_{dataset_name}.txt")
raw_output_file = open(raw_output_path, "w", encoding="utf-8")



# --- helper: check if track is valid ---
def is_valid_track(sample: str) -> bool:
    """Check if a decoded track has at least B, one coordinate, and E in correct order."""
    lines = [line.strip() for line in sample.splitlines() if line.strip()]
    if not lines:
        return False

    try:
        start_idx = lines.index("B")
        end_idx = lines.index("E", start_idx + 1)
    except ValueError:
        return False  # no B or no E

    # there should be at least one coordinate line between B and E
    coords_between = lines[start_idx+1:end_idx]
    if len(coords_between) == 0:
        return False

    return True


# --- generation loop with batching ---
batch_size = 8  # adjust depending on GPU memory

with torch.no_grad():
    with ctx, open(output_path, "w", encoding="utf-8") as f:
        # write CSV header
        f.write("tframe,y,x,track_no\n")

        track_no = 1
        e_count = 0
        invalid_count = 0
        bad_count = 0
        target_track_count = 50  # Stop after generating this many valid tracks
    
        # Start timer
        start_time = time.time()

        while e_count < target_track_count:
            # prepare a batch of identical prompts
            generated = x.clone().repeat(batch_size, 1)
            finished = [False] * batch_size  # track which sequences are done

            for _ in range(max_new_tokens):  # max length safeguard
                block_size = model.config.block_size
                input_ids = generated[:, -block_size:]
                logits = model(input_ids)[0][:, -1, :] / temperature

                if top_k:
                    topk = torch.topk(logits, top_k)
                    filter_mask = logits < topk.values[:, -1].unsqueeze(-1)
                    logits[filter_mask] = -float('Inf')

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)

                # mark finished sequences
                for i in range(batch_size):
                    if not finished[i] and next_token[i].item() == e_token:
                        finished[i] = True

                if all(finished):
                    break  # stop once all batch samples hit E

            # decode and validate each sample in the batch
            for i in range(batch_size):
                sample = decode(generated[i].tolist())
                raw_output_file.write(sample + "\n\n")  # log raw sample

                if is_valid_track(sample):
                    e_count += 1
                    print(f"✅ Valid track {e_count}/{target_track_count} complete.")

                    # parse and write CSV rows
                    lines = [line.strip() for line in sample.splitlines() if line.strip()]
                    for tframe, line in enumerate(lines):
                        if line in ["B", "E"]:
                            continue

                        parts = line.split()
                        if len(parts) != 2 or not all(p.lstrip("-").isdigit() for p in parts):
                            bad_count += 1
                            continue

                        y, x_coord = parts
                        f.write(f"{tframe},{y},{x_coord},{track_no}\n")

                    track_no += 1

                    if e_count >= target_track_count:
                        break  # stop immediately once we hit target
                else:
                    invalid_count += 1
                    print(f"❌ Invalid track skipped. Total skipped: {invalid_count}")

        # End timer
        end_time = time.time()
        elapsed_time = end_time - start_time
        minutes, seconds = divmod(elapsed_time, 60) 

        # Print summary 
        print(f"Total tracks generated: {e_count + invalid_count}")
        print(f"Valid tracks: {e_count}")
        print(f"Invalid/skipped tracks: {invalid_count}")
        print(f"Malformed coordinate lines: {bad_count}")
        print(f"Sample generation took {int(minutes)} minutes and {seconds:.2f} seconds.")