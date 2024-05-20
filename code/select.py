import os  
import shutil  
  
# 假设txt文件路径和语音集目录如下  
txt_file_path = 'F:\\thesis\data_set\LA\ASVspoof2019_LA_train\\flac\ASVspoof2019.LA.cm.train.trn.txt'  
voice_set_directory = 'F:\\thesis\data_set\LA\ASVspoof2019_LA_train\\flac'  
  
# 创建文件夹列表  
folder_names = ['bonafide', 'A01 spoof', 'A02 spoof', 'A03 spoof', 'A04 spoof', 'A05 spoof', 'A06 spoof']  
  
# 确保所有目标文件夹都存在  
for folder_name in folder_names:  
    folder_path = os.path.join(voice_set_directory, folder_name)  
    if not os.path.exists(folder_path):  
        os.makedirs(folder_path)  
  
# 读取txt文件并移动文件  
with open(txt_file_path, 'r') as txt_file:  
    for line in txt_file:  
        # 去除行尾的换行符  
        line = line.strip()  
          
        # 分割文件名和类型  
        parts = line.split()  
        file_name = parts[1]  # 假设第二个元素是文件名  
        type_part = ' '.join(parts[2:])  # 将剩余部分组合起来作为类型  
        print(type_part)
       # 查找类型部分是否以'--'或'-'开头，并去除开头的'--'或'-'  
        if type_part.startswith('- - '):  
            file_type = type_part[4:].strip()  # 去除开头的'--'和可能的尾随空格  
        elif type_part.startswith('- '):  
            file_type = type_part[2:].strip()  # 去除开头的'-'和可能的尾随空格  
        else:  
            # 如果不是以'--'或'-'开头，则提取最后一个非空白元素作为类型  
            file_type = type_part.strip().split()[-1]  
        print(file_type)
        # 构造文件的完整路径（忽略LA_四位数字部分）  
        source_path = os.path.join(voice_set_directory, file_name+ '.flac') 
        print(source_path)
        # 检查文件是否存在  
        if os.path.exists(source_path):  
            # 构造目标文件夹的完整路径  
            destination_path = os.path.join(voice_set_directory, file_type, file_name+ '.flac' ) 
            print(destination_path)
            # 移动文件到目标文件夹  
            shutil.move(source_path, destination_path)  
            print(f"Moved {file_name} to {file_type} folder.")  
        else:  
            print(f"File {file_name} not found.")  
  
print("File sorting completed.")