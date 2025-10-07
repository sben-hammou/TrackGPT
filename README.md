# TrackGPT 

This repository contains the code for training a GPT model on cell tracking data. The model is based on Andrej Karpathy's NanoGPT and has been adapted to work with cell tracking datasets. The original repository contains datasets from a preprint paper on cell tracking (available here: https://www.biorxiv.org/content/10.1101/2024.10.21.618803v1 ), which cannot be shared here. However, one cell tracking dataset is included for demonstration purposes.



## How to train the model

To train the model, follow these steps:

1. Modify prepare.py file to set the correct directory path for your dataset.

2. Create a config file and change the out-dir, wandb-project, and dataset parameters to match your dataset name. In this example, the datset is called train10-tcell. 

3. Run the prepare.py script as follows: 

```sh
python data/train-tcell/prepare.py
``` 

4. Start the training process with the following command:

```sh
python train.py config/train_train10-tcell.py # add the following if running on CPU: --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0  
```

5. Once the training is complete, you can generate a sample from the trained model using: 

```sh
python sample.py --out_dir=out-train10-tcell # add the following if running on CPU: --device=cpu
``` 

## See lab_journal.md for the detailed lab journal of my internship project.