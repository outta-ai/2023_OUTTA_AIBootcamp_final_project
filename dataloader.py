from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

def get_dataloader(args, dataset, is_train=True):
    if is_train:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    dataloader = DataLoader(dataset=dataset, sampler=sampler, batch_size=args.batch_size)
    return dataloader