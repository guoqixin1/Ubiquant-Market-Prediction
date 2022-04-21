import json
import csv
from collections import defaultdict


# split file for memory reason
file_list = [
    '/cfs/cfs-ps2i4uy9/qixiguo/ubi/data/train_0.csv',
    '/cfs/cfs-ps2i4uy9/qixiguo/ubi/data/train_1.csv',
    '/cfs/cfs-ps2i4uy9/qixiguo/ubi/data/train_2.csv',
    '/cfs/cfs-ps2i4uy9/qixiguo/ubi/data/train_3.csv',
    '/cfs/cfs-ps2i4uy9/qixiguo/ubi/data/train_4.csv',
    '/cfs/cfs-ps2i4uy9/qixiguo/ubi/data/train_5.csv',
    '/cfs/cfs-ps2i4uy9/qixiguo/ubi/data/train_6.csv',
    '/cfs/cfs-ps2i4uy9/qixiguo/ubi/data/train_7.csv',
    '/cfs/cfs-ps2i4uy9/qixiguo/ubi/data/train_8.csv',
    '/cfs/cfs-ps2i4uy9/qixiguo/ubi/data/train_9.csv'
]
i = 0
# group data by invest names
for file_name in file_list:
    invest2data = defaultdict(list)
    with open(file_name)as f:
        csv_reader = csv.reader(f)
        for items in csv_reader:
            invest2data[items[2]].append(items)
    with open('/cfs/cfs-ps2i4uy9/qixiguo/ubi/data/invest2data_' + str(i) + '.json', 'w')as f:
        json.dump(invest2data, f, ensure_ascii=False)
    i += 1
