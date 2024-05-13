import pandas as pd
import numpy as np

def preprocess(file_path_name, output_path_name):
    """
    Preprocess the data and fill missing values with forward fill values
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
    numeric_columns = ['Active_Energy_Delivered_Received', 'Current_Phase_Average', 'Wind_Speed',
                       'Weather_Temperature_Celsius', 'Weather_Relative_Humidity',
                       'Global_Horizontal_Radiation', 'Diffuse_Horizontal_Radiation', 'Wind_Direction',
                       'Weather_Daily_Rainfall', 'Radiation_Global_Tilted', 'Radiation_Diffuse_Tilted', 'Active_Power']

    # 明确保空字符串被识别为NaN
    data.replace('', np.nan, inplace=True)  # 直接在整个DataFrame上操作，避免了循环

    # 使用上一行的值填充缺失值
    data[numeric_columns] = data[numeric_columns].ffill()

    # 计算数值列的均值和标准差（注意：填充后，某些统计结果可能因数据变化而不同）
    summary = data[numeric_columns].describe()

    print(summary)

    # 保存处理后的数据到新CSV
    output_path = output_path_name  # 换为新文件路径
    data.to_csv(output_path, index=False)

    print("数据已成功保存到:", output_path)

if __name__ == '__main__':
    preprocess('../data/1A-Real-Trina.csv', '../data/1A-Real-Trina-new.csv')