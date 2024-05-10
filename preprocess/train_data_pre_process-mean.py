import pandas as pd
import numpy as np

def preprocess(file_path_name, output_path_name):
    """
    Preprocess the data, fill missing values with mean and set the Active_Power >= 0
    :param file_path_name:
    :param output_path_name:
    :return:
    """
    # 设置显示选项以显示所有列，不限制宽
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    # 读取CSV文件
    file_path = file_path_name  # 替换为您的CSV文件路径
    data = pd.read_csv(file_path)

    # 定义数值列列表，根据实际列名调整
    # Active_Energy_Delivered_Received,Current_Phase_Average,Wind_Speed,Weather_Temperature_Celsius,
    # Weather_Relative_Humidity,Global_Horizontal_Radiation,Diffuse_Horizontal_Radiation,Wind_Direction,
    # Weather_Daily_Rainfall,Radiation_Global_Tilted,Radiation_Diffuse_Tilted,Active_Power
    numeric_columns = ['Active_Energy_Delivered_Received', 'Current_Phase_Average', 'Wind_Speed', 'Weather_Temperature_Celsius',
                       'Weather_Relative_Humidity', 'Global_Horizontal_Radiation', 'Diffuse_Horizontal_Radiation', 'Wind_Direction',
                       'Weather_Daily_Rainfall', 'Radiation_Global_Tilted', 'Radiation_Diffuse_Tilted', 'Active_Power']

    # 明确保空字符串被识别为NaN
    for col in data.columns:
        data[col] = data[col].replace('', np.nan)

    # 使用均值填充缺失值
    for col in numeric_columns:
        print("开始替代！！！")
        data[col].fillna(data[col].mean(), inplace=True)

    # 将Active_Power列中的负数置为0
    data['Active_Power'] = data['Active_Power'].clip(lower=0)

    # 计算数值列的均值和标准差
    summary = data[numeric_columns].describe()

    print(summary)

    # 保存处理后的数据到新CSV
    output_path = output_path_name  # 换为新文件路径
    data.to_csv(output_path, index=False)

    print("数据已成功保存到:", output_path)

if __name__ == '__main__':
    preprocess('data/1A-Real-Trina.csv', 'data/1A-Real-Trina-new.csv')
    # preprocess('data/1A-Trina-Test.csv', 'data/1A-Trina-Test-new.csv')
