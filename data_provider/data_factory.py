from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar, Dataset_PEMS, \
    Dataset_Pred, Dataset_Dpos
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Solar': Dataset_Solar,
    'PEMS': Dataset_PEMS,
    'custom': Dataset_Custom,
    'dpos': Dataset_Dpos,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        # data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        target_channel=3
        # freq=freq,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader


# import torch
# from torch.utils.data import DataLoader
# from data_provider.data_loader import Dataset_Dpos

# def data_provider(args, flag):
#     """
#     根据任务类型选择合适的数据集。
#     你自定义的数据集：使用 Dataset_Dpos，适配 .mat 数据。
#     """
#     if args.data == 'dpos':  # 你设置的 data 参数为 'dpos'
#         normalize = True if flag == 'train' else False

#         dataset = Dataset_Dpos(
#             data_dir=args.root_path,
#             target_channel=0,              # 可修改为参数 args.target_channel
#             L=args.seq_len,
#             S=args.pred_len,
#             mode=flag,
#             normalize=normalize
#         )

#         if not normalize:
#             # 如果不是 train，则需要设置标准化参数
#             # 这里假设 train 集合已经初始化完成
#             train_dataset = Dataset_Dpos(
#                 data_dir=args.root_path,
#                 target_channel=0,
#                 L=args.seq_len,
#                 S=args.pred_len,
#                 mode='train',
#                 normalize=True
#             )
#             dataset.set_norm_stats(train_dataset.mean, train_dataset.std)

#         dataloader = DataLoader(
#             dataset,
#             batch_size=args.batch_size,
#             shuffle=(flag == 'train'),
#             num_workers=args.num_workers
#         )

#         return dataset, dataloader

#     else:
#         raise ValueError(f"Unsupported dataset: {args.data}")

