# -*- coding: cp1251 -*-
import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize



def clearing_data(data_file):
    data = pd.read_csv(data_file, error_bad_lines=False, index_col=False)

    # replacing missing data
    data.replace('Не указано', np.NaN,  inplace=True)
    data.replace('-', np.NaN, inplace=True)
    data['Цель испытаний'] = data['Цель испытаний'].replace(np.NaN, 'Не обозначено')
    data['Цель испытаний'] = data['Цель испытаний'].replace('Контрольные испытания', 'Не обозначено')
    data['Электроустановка'] = data['Электроустановка'].replace('Старая - ', 'Не обозначено')
    data['Электроустановка'] = data['Электроустановка'].replace('Новая - ', 'Не обозначено')
    data['Электроустановка'] = data['Электроустановка'].replace(' - Удовлетворительно', 'Не обозначено')
    data['Электроустановка'] = data['Электроустановка'].replace(' - ', 'Не обозначено')

    # droping multiple missing row

    data = data.dropna(axis=0, thresh=18)
    columns = ['Цель испытаний','Электроустановка','Количество щитов','Наличие цепи: кол-во точек','Сопротивление изоляции: кол-во линий','Фаза-нуль: количество линий','УЗО: Количество УЗО','Автоматы: Количество Автоматов','Заземлителей: Кол-во ЗУ','Молниезащита: Кол-во МЗ','Затраченное время общее (минуты)','Среднее время заполнения линии (минуты)','Затраченное время Планшетное (минуты)','Затраченное время Вебверсия (минуты)','Затраченное эффективное время общее (минуты)','Затраченное эффективное время Планшетное (минуты)','Затраченное эффективное время Вебверсия (минуты)','Среднее время межу щитами общее (минуты)','Среднее время межу щитами Планшетное (минуты)','Среднее время межу щитами Вебверсия (минуты)', 'Инженер']
    df = data[columns]

    # Naming part of data
    column_time = ['Затраченное время общее (минуты)','Среднее время заполнения линии (минуты)','Затраченное время Планшетное (минуты)','Затраченное время Вебверсия (минуты)','Затраченное эффективное время общее (минуты)','Затраченное эффективное время Планшетное (минуты)','Затраченное эффективное время Вебверсия (минуты)','Среднее время межу щитами общее (минуты)','Среднее время межу щитами Планшетное (минуты)','Среднее время межу щитами Вебверсия (минуты)']
    column_automat =[
        'Наличие цепи: кол-во точек',
        'Количество щитов',
        'Сопротивление изоляции: кол-во линий',
        'Фаза-нуль: количество линий',
        'Автоматы: Количество Автоматов',
        'УЗО: Количество УЗО',
        'Заземлителей: Кол-во ЗУ',
        'Молниезащита: Кол-во МЗ'
    ]

    cat_cols = ['Электроустановка','Цель испытаний','Инженер']

    # filling nan data
    df[column_automat] = df[column_automat].fillna(0)
    df[column_time] = df[column_time].fillna(0)

    # changing type of data
    df[column_automat] = df[column_automat].astype('int32')
    df[column_time] = df[column_time].astype('float32')

    # highligting more positive column for prediction
    num_cols = [
        'Наличие цепи: кол-во точек',
        'Количество щитов',
        'Сопротивление изоляции: кол-во линий',
        'Фаза-нуль: количество линий',
        'Автоматы: Количество Автоматов',
        'УЗО: Количество УЗО',
        'Заземлителей: Кол-во ЗУ',
        'Молниезащита: Кол-во МЗ',
        # 'Среднее время заполнения линии (минуты)',
        # 'Среднее время межу щитами общее (минуты)',
        # 'Затраченное время общее (минуты)'
    ]

    target_col= ['Затраченное эффективное время общее (минуты)']

    df[target_col] = df[target_col].astype('float32')

    # target data
    y = abs(df[target_col])

    #droping anomaling for data
    df = df.drop(df.loc[abs(df['Затраченное эффективное время общее (минуты)']) > 1000].index)
    df = df.drop(df.loc[abs(df['Количество щитов']) == 0].index)
    df = df.drop(df.loc[abs(df['Сопротивление изоляции: кол-во линий']) == 0].index)
    df = df.drop(df.loc[abs(df['Фаза-нуль: количество линий']) == 0].index)
    df = df.drop(df.loc[df['Инженер'].str.contains('KozhinDev')].index)
    df = df.drop(df.loc[df['Инженер'].str.contains('совместный отчет')].index)
    df = df.drop(df.loc[df['Инженер'].str.contains('anatoliymladensky@gmail.com')].index)
    df = df.drop(df.loc[df['Цель испытаний'].str.contains('Не обозначено')].index)
    df = df.drop(df.loc[df['Электроустановка'].str.contains('Не обозначено')].index)



    # df = df.drop(df.loc[abs(df['Электроустановка']) == 'Старая - Не удовлетворительно' and (df['Цель испытаний'] == 'Эксплуатационные']).index)
    df = df.drop(df.loc[abs(df['Затраченное эффективное время общее (минуты)']) > 1000].index)

    #Enable to get a normal distribution over the target values and use linear algorithms
    df = df.drop(df.loc[abs(df['Сопротивление изоляции: кол-во линий']) > 100].index)
    df = df.drop(df.loc[abs(df['Фаза-нуль: количество линий']) > 100].index)

    #Enable a normal distribution for all values
    # df = df.drop(df.loc[abs(df['Наличие цепи: кол-во точек']) > 100].index)
    # df = df.drop(df.loc[abs(df['Количество щитов']) > 100].index)
    # df = df.drop(df.loc[abs(df['Автоматы: Количество Автоматов']) > 100].index)
    # df = df.drop(df.loc[abs(df['УЗО: Количество УЗО']) > 100].index)
    # df = df.drop(df.loc[abs(df['Заземлителей: Кол-во ЗУ']) > 100].index)
    # df = df.drop(df.loc[abs(df['Молниезащита: Кол-во МЗ']) > 100].index)

    # concatenated columns numerated col and categorial
    X_data = pd.concat([df[num_cols], df[cat_cols]], axis=1)

    # math.stat operation winsorize
    #df[num_cols] = winsorize(df[num_cols], limits=[0.1,0.2])

    #Optimal data volume
    dummy_features = pd.get_dummies(df[cat_cols])
    X = pd.concat([df[num_cols], dummy_features], axis=1)
    X_three = pd.concat([df[num_cols], df[cat_cols]], axis = 1)

    y = abs(df[target_col])
    return X, y

# list1 = "/Users/mac/Desktop/fast_api_xgb/reports-9.csv"
#
#
# x = clearing_data(list1)
#
#
# print(x)

