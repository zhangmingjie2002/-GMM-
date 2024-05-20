select.py：实现了根据ASVspoof数据库中提供的ASVspoof2019.LA.cm.dev.trl.txt和ASVspoof2019.LA.cm.train.trn.txt标记文件根据标记将训练集与开发集分入bonafide、fake、A01 spoof、A02 spoof、A03 spoof、A04 spoof、A05 spoof、A06 spoof这八个文件夹
select_train.py：实现了根据ASVspoof数据库中提供的ASVspoof2019.LA.cm.eval.trl.txt标记文件根据标记将评估集分入bonafide、fake这两个文件夹
./source：包含了模型训练、特征提取、数据选择算法实现的.py代码