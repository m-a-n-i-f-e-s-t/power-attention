import os
import argparse

import numpy as np
import tiktoken
from datasets import load_dataset  # huggingface datasets
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare news dataset for training')
    parser.add_argument('--destination', type=str, 
                       default=os.path.expanduser('~/mai_datasets/news'),
                       help='Folder where final dataset will be saved (default: ~/mai_datasets/news)')
    parser.add_argument('--num_proc', type=int, 
                       default=os.cpu_count() // 2,
                       help='Number of workers for processing (default: cpu_count // 2)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # folder where final dataset will live
    os.makedirs(args.destination, exist_ok=True)
    destination_folder = args.destination
    num_proc = args.num_proc

    enc = tiktoken.get_encoding('gpt2')
    dataset = load_dataset('permutans/fineweb-bbc-news', 'CC-MAIN-2013-20', num_proc=num_proc)
    split_dataset = dataset['train'].train_test_split(test_size=0.01, seed=2357, shuffle=True)
    split_dataset['heldout'] = split_dataset.pop('test') # rename the test split to heldout

    def process(example):
        ids = enc.encode_ordinary(example['text'])
        ids.append(enc.eot_token)
        out = {'ids': ids, 'len': len(ids)}
        return out

    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc='tokenizing the splits',
        num_proc=num_proc,
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(destination_folder, f'{split}.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()