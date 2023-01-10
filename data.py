# -*- coding: cp1251 -*-
import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize



def clearing_data(data_file):
    data = pd.read_csv(data_file, error_bad_lines=False, index_col=False)

    # replacing missing data
    data.replace('�� �������', np.NaN,  inplace=True)
    data.replace('-', np.NaN, inplace=True)
    data['���� ���������'] = data['���� ���������'].replace(np.NaN, '�� ����������')
    data['���� ���������'] = data['���� ���������'].replace('����������� ���������', '�� ����������')
    data['����������������'] = data['����������������'].replace('������ - ', '�� ����������')
    data['����������������'] = data['����������������'].replace('����� - ', '�� ����������')
    data['����������������'] = data['����������������'].replace(' - �����������������', '�� ����������')
    data['����������������'] = data['����������������'].replace(' - ', '�� ����������')

    # droping multiple missing row

    data = data.dropna(axis=0, thresh=18)
    columns = ['���� ���������','����������������','���������� �����','������� ����: ���-�� �����','������������� ��������: ���-�� �����','����-����: ���������� �����','���: ���������� ���','��������: ���������� ���������','������������: ���-�� ��','������������: ���-�� ��','����������� ����� ����� (������)','������� ����� ���������� ����� (������)','����������� ����� ���������� (������)','����������� ����� ��������� (������)','����������� ����������� ����� ����� (������)','����������� ����������� ����� ���������� (������)','����������� ����������� ����� ��������� (������)','������� ����� ���� ������ ����� (������)','������� ����� ���� ������ ���������� (������)','������� ����� ���� ������ ��������� (������)', '�������']
    df = data[columns]

    # Naming part of data
    column_time = ['����������� ����� ����� (������)','������� ����� ���������� ����� (������)','����������� ����� ���������� (������)','����������� ����� ��������� (������)','����������� ����������� ����� ����� (������)','����������� ����������� ����� ���������� (������)','����������� ����������� ����� ��������� (������)','������� ����� ���� ������ ����� (������)','������� ����� ���� ������ ���������� (������)','������� ����� ���� ������ ��������� (������)']
    column_automat =[
        '������� ����: ���-�� �����',
        '���������� �����',
        '������������� ��������: ���-�� �����',
        '����-����: ���������� �����',
        '��������: ���������� ���������',
        '���: ���������� ���',
        '������������: ���-�� ��',
        '������������: ���-�� ��'
    ]

    cat_cols = ['����������������','���� ���������','�������']

    # filling nan data
    df[column_automat] = df[column_automat].fillna(0)
    df[column_time] = df[column_time].fillna(0)

    # changing type of data
    df[column_automat] = df[column_automat].astype('int32')
    df[column_time] = df[column_time].astype('float32')

    # highligting more positive column for prediction
    num_cols = [
        '������� ����: ���-�� �����',
        '���������� �����',
        '������������� ��������: ���-�� �����',
        '����-����: ���������� �����',
        '��������: ���������� ���������',
        '���: ���������� ���',
        '������������: ���-�� ��',
        '������������: ���-�� ��',
        # '������� ����� ���������� ����� (������)',
        # '������� ����� ���� ������ ����� (������)',
        # '����������� ����� ����� (������)'
    ]

    target_col= ['����������� ����������� ����� ����� (������)']

    df[target_col] = df[target_col].astype('float32')

    # target data
    y = abs(df[target_col])

    #droping anomaling for data
    df = df.drop(df.loc[abs(df['����������� ����������� ����� ����� (������)']) > 1000].index)
    df = df.drop(df.loc[abs(df['���������� �����']) == 0].index)
    df = df.drop(df.loc[abs(df['������������� ��������: ���-�� �����']) == 0].index)
    df = df.drop(df.loc[abs(df['����-����: ���������� �����']) == 0].index)
    df = df.drop(df.loc[df['�������'].str.contains('KozhinDev')].index)
    df = df.drop(df.loc[df['�������'].str.contains('���������� �����')].index)
    df = df.drop(df.loc[df['�������'].str.contains('anatoliymladensky@gmail.com')].index)
    df = df.drop(df.loc[df['���� ���������'].str.contains('�� ����������')].index)
    df = df.drop(df.loc[df['����������������'].str.contains('�� ����������')].index)



    # df = df.drop(df.loc[abs(df['����������������']) == '������ - �� �����������������' and (df['���� ���������'] == '����������������']).index)
    df = df.drop(df.loc[abs(df['����������� ����������� ����� ����� (������)']) > 1000].index)

    #Enable to get a normal distribution over the target values and use linear algorithms
    df = df.drop(df.loc[abs(df['������������� ��������: ���-�� �����']) > 100].index)
    df = df.drop(df.loc[abs(df['����-����: ���������� �����']) > 100].index)

    #Enable a normal distribution for all values
    # df = df.drop(df.loc[abs(df['������� ����: ���-�� �����']) > 100].index)
    # df = df.drop(df.loc[abs(df['���������� �����']) > 100].index)
    # df = df.drop(df.loc[abs(df['��������: ���������� ���������']) > 100].index)
    # df = df.drop(df.loc[abs(df['���: ���������� ���']) > 100].index)
    # df = df.drop(df.loc[abs(df['������������: ���-�� ��']) > 100].index)
    # df = df.drop(df.loc[abs(df['������������: ���-�� ��']) > 100].index)

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

