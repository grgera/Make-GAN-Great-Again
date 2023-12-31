import torch
from math import ceil
import numpy as np

def prepare_generator_batch(oracle, gen, batch_size, gpu=False):
    """
    Returns samples (a batch) sampled from generator. 

    Returns: target, target_lens, cond, cond_lens, end_of_dataset
        - target, cond: batch_size x seq_len
        - target_lens, cond_lens: batch_size
    """
    pos_samples, pos_lens, cond_ids, end_of_dataset = oracle.sample(batch_size, gpu=gpu)
    batch_size = len(cond_ids) 
    cond_samples, cond_lens = oracle.fetch_cond_samples(cond_ids, gpu=gpu)
    target, target_lens = gen.sample(cond_samples, gpu=gpu)

    target = trim_trailing_paddings(target, target_lens)

    # Put to GPU
    if gpu:
        target = target.cuda()
        target_lens = target_lens.cuda()
        pos_samples = pos_samples.cuda()
        pos_lens = pos_lens.cuda()
        cond_samples = cond_samples.cuda()
        cond_lens = cond_lens.cuda()

    return target, target_lens, pos_samples, pos_lens, cond_samples, cond_lens, end_of_dataset

def prepare_discriminator_data(oracle, gen, batch_size, is_val=False, on_cpu=True, gpu=False, gpu_limit=None):
    """
    Takes positive (target), negative (generator), and condition sample generators/loaders 
    to prepare inp and target samples for discriminator.
    Put final tensor on cpu if on_cpu is set.
    Use gpu to generate data if gpu is set, but only gpu_limit size of data can be on GPU at a time.

    Returns: inp, inp_lens, cond, cond_lens, target, end_of_dataset
        - inp, cond: batch_size x seq_len
        - inp_lens, cond_lens: batch_size
        - target: batch_size (boolean 1/0)
    """
    half_batch_size = int(batch_size / 2)
    pad_token = oracle.pad_token

    # Prepare pos, cond samples from oracle
    pos_samples, pos_lens, cond_ids, end_of_dataset = oracle.sample(half_batch_size, is_val=is_val, gpu=gpu)
    half_batch_size = len(cond_ids) 
    batch_size = half_batch_size * 2
    cond_samples, cond_lens = oracle.fetch_cond_samples(cond_ids, gpu=gpu)

    # Prepare neg samples generated by generator from cond
    neg_samples = torch.ones(half_batch_size, gen.max_seq_len).long() * gen.pad_token
    neg_lens = torch.ones(half_batch_size).long() * gen.pad_token
    if not on_cpu and gpu:
        neg_samples = neg_samples.cuda()
        neg_lens = neg_lens.cuda()

    for i in range(0, half_batch_size, gpu_limit): 
        # Generate data with GPU if gpu set
        neg_samples_temp, neg_lens_temp = gen.sample(cond_samples[i:i+gpu_limit], gpu=gpu)

        # Keep data in CPU
        if on_cpu:
            neg_samples_temp, neg_lens_temp = neg_samples_temp.cpu(), neg_lens_temp.cpu()

        # Put together
        temp_batch_size, temp_seq_len = neg_samples_temp.shape
        neg_samples[i:i+temp_batch_size, :temp_seq_len] = neg_samples_temp
        neg_lens[i:i+temp_batch_size] = neg_lens_temp

    neg_samples = trim_trailing_paddings(neg_samples, neg_lens)

    if on_cpu: 
        pos_samples, pos_lens, cond_samples, cond_lens = pos_samples.cpu(), pos_lens.cpu(), cond_samples.cpu(), cond_lens.cpu()

    _, pos_seq_len = pos_samples.shape
    _, neg_seq_len = neg_samples.shape
    _, cond_seq_len = cond_samples.shape

    # Concat
    inp, inp_lens = cat_samples(pos_samples, pos_lens, neg_samples, neg_lens, pad_token)
    cond, cond_lens = cat_samples(cond_samples, cond_lens, cond_samples, cond_lens, pad_token)

    # Construct target
    target = torch.ones(batch_size)
    target[half_batch_size:] = 0

    # Shuffle
    perm = torch.randperm(batch_size)
    inp, inp_lens, cond, cond_lens, target = inp[perm], inp_lens[perm], cond[perm], cond_lens[perm], target[perm]

    return inp, inp_lens, cond, cond_lens, target, end_of_dataset

def batchwise_sample(gen, num_samples, batch_size):
    """
    NOT USED.
    Sample num_samples samples batch_size samples at a time from gen.
    Does not require gpu since gen.sample() takes care of that.
    """
    samples = []
    for i in range(int(ceil(num_samples / float(batch_size)))):
        samples.append(gen.sample(batch_size))

    return torch.cat(samples, 0)[:num_samples]

def batchwise_oracle_nll(gen, oracle, num_samples, batch_size, max_seq_len, start_letter=0, gpu=False):
    """
    NOT USED.
    """
    s = batchwise_sample(gen, num_samples, batch_size)
    oracle_nll = 0
    for i in range(0, num_samples, batch_size):
        inp, target = prepare_generator_batch(s[i:i+batch_size], start_letter, gpu)
        oracle_loss = oracle.batchNLLLoss(inp, target) / max_seq_len
        oracle_nll += oracle_loss.data.item()

    return oracle_nll / (num_samples / batch_size)

def pad_samples(samples, pad_token):
    """
    Pad samples of variable lengths with pad_token.

    Output: padded_samples, lens
        - padded_samples: num_samples x max_seq_len
        - lens: num_samples
    """
    num_samples = len(samples)

    # Get length info
    lens = np.array([len(x) for x in samples])
    max_len = max(lens)

    # Pad
    padded_samples = np.ones((num_samples, max_len), dtype=np.int) * pad_token

    for i, seq in enumerate(samples):
        padded_samples[i, :lens[i]] = seq

    padded_samples = torch.LongTensor(padded_samples)
    lens = torch.LongTensor(lens)
    return padded_samples, lens

def cat_samples(samples_1, lens_1, samples_2, lens_2, pad_token):
    """
    Concat two batches of samples of variable sizes, and pad with pad_token.

    Inputs:
        - samples_1, samples_2: num_samples x seq_len
        - lens_1, lens_2: num_samples
    Outputs: samples, lens
        - samples: num_samples x seq_len
        - lens: num_samples
    """
    num_samples_1, num_samples_2 = len(samples_1), len(samples_2)
    seq_len_1, seq_len_2 = samples_1.shape[1], samples_2.shape[1]
    max_len = max(seq_len_1, seq_len_2)

    # Concat & pad
    samples = np.ones((num_samples_1 + num_samples_2, max_len), dtype=np.int) * pad_token
    samples[:num_samples_1, :seq_len_1] = samples_1
    samples[num_samples_1:, :seq_len_2] = samples_2
    samples = torch.LongTensor(samples)
    lens = torch.cat([lens_1, lens_2])

    return samples, lens

def trim_trailing_paddings(samples, sample_lens):
    """
    Trim trailing pad_tokens to align all rows with the max sequence.

    Inputs:
        - samples: batch_size x seq_len
        - sample_lens: batch_size
    Outputs: samples
        - samples: batch_size x seq_len
    """
    max_seq_len = torch.max(sample_lens)
    return samples[:, :max_seq_len]

def sort_sample_by_len(samples, lens):
    """
    Sort samples by length in descending order.

    Inputs:
        - samples: batch_size x seq_len
        - lens: batch_size

    Outputs:
        - samples: batch_size x seq_len
        - lens: batch_size
        - sort_idx: batch_size
    """
    lens, sort_idx = torch.sort(lens, descending=True)
    samples = samples[sort_idx]
    return samples, lens, sort_idx