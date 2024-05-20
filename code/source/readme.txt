extract_cqcc.py：提取CQCC特征值的接口函数实现
extract_lfcc.py：提取LFCC特征值的接口函数实现
extract_mfcc.py：提取MFCC特征值的接口函数实现
model_train.py：训练32维真实、伪造gmm模型代码实现
model_test.py：根据joblib保存的gmm模型计算等错误率并绘制FAR、FFR与阈值$\theta$的图像
./data_select：包含了实现的数据选择算法相关代码
注意：在实际应用的过程中，需要根据实际提取的特征值类型，对model_train.py和model_test.py中导包模块的导包名称进行相应修改