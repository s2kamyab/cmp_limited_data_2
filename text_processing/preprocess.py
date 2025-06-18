import os
from datetime import timedelta
from datetime import datetime
import pandas as pd
from datetime import datetime


# 反推相对时间
def convert_to_utc(time_str):
    # 检查并去除时区缩写
    if " EDT" in time_str:
        time_str_cleaned = time_str.replace(" EDT", "")
        offset = timedelta(hours=-4)
    elif " EST" in time_str:
        time_str_cleaned = time_str.replace(" EST", "")
        offset = timedelta(hours=-5)
    else:
        # 默认为0时差，对于只有日期的情况不调整时区
        offset = timedelta(hours=0)
        time_str_cleaned = time_str

    # 尝试不同的日期时间格式
    formats = [
        '%B %d, %Y — %I:%M %p',  # "September 12, 2023 — 06:15 pm"
        '%b %d, %Y %I:%M%p',  # "Nov 14, 2023 7:35AM"
        '%d-%b-%y',  # "6-Jan-22"
        '%Y-%m-%d',  # "2021-4-5"
        '%Y/%m/%d',  # "2021/4/5"
        '%b %d, %Y'  # "DEC 7, 2023"
    ]

    for fmt in formats:
        try:
            # 尝试解析日期和时间
            dt = datetime.strptime(time_str_cleaned, fmt)
            # 如果格式只包含日期，不包含具体时间，则不应用时区调整
            if fmt == '%d-%b-%y':
                offset = timedelta(hours=0)

            # 调整为UTC时间
            dt_utc = dt + offset

            return dt_utc.strftime('%Y-%m-%d %H:%M:%S UTC')
        except ValueError:
            continue

    # 如果所有格式都不匹配，返回错误信息
    return "Invalid date format"


def date_inte(folder_path, saving_path):
    txt_files = [file for file in os.listdir(folder_path) if file.endswith('.txt')]
    df = pd.DataFrame()# target_date_str = date.strftime('%d_%m_%Y')
    for file in txt_files:
        date_obj = datetime.strptime(file[-14:-4], '%d_%m_%Y')
        # if target_date_str in file:
        with open(folder_path + file, 'r', encoding='latin-1') as f:
            text = f.read()
        new_row = {'Date': date_obj, 'text': text}
        if len(df) == 0:
            df = pd.DataFrame({'Date':[date_obj], 'text':[text]})
        else:
            df.loc[len(df)] = new_row#df.append(new_row, ignore_index=True)
    # for txt_file in txt_files:
    #     print('Starting: ' + txt_file)
    #     file_path = os.path.join(folder_path, txt_file)
    #     # 使用pandas的read_csv函数读取CSV文件
    #     df = pd.read_csv(file_path, on_bad_lines="warn")
        # df.columns = df.columns.str.capitalize()
        # if 'Datetime' in df.columns:
        #     df.rename(columns={'Datetime': 'Date'}, inplace=True)
        # # 应用转换函数
        # print(df["Date"])
    # df['Date'] = df['Date'].apply(convert_to_utc)
    # print(df["Date"])
    # # 将Date列转换为日期时间格式
    df['Date'] = pd.to_datetime(df['Date'])#, utc=True)
    # 按照Date列降序排序
    df = df.sort_values(by='Date', ascending=False)
    # 输出结果
    # print(df)

    df.to_csv(os.path.join(saving_path, 'soshianest_news.csv'), index=False)
    print('Done: ' + 'soshianest_news.csv')


if __name__ == "__main__":
    news_folder_path = "C:\\Users\\shima\\Documents\\Postdoc_Uvic\\Paper1\\Data\\Soshianest\\news\\all_reports_csv\\"
    news_saving_path = "C:\\Users\\shima\\Documents\\Postdoc_Uvic\\Paper1\\Data\\Soshianest\\news\\finspd_processed_news\\"

    stock_folder_path = "C:\\Users\\shima\\Documents\\Postdoc_Uvic\\Paper1\\Data\\Soshianest\\separated_dataset\\"
    stock_saving_path = "C:\\Users\\shima\\Documents\\Postdoc_Uvic\\Paper1\\Data\\Soshianest\\separated_dataset_processed\\"

    date_inte(news_folder_path, news_saving_path)
    # date_inte(stock_folder_path, stock_saving_path)

