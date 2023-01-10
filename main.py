from fastapi import FastAPI
import pandas as pd
import joblib
import model
import data
from joblib import dump


app = FastAPI()

def getParamsForModel(params):
    engineers = {
        "aksyonov_sergey":"Инженер_Аксенов Сергей",
        "alexey_ryazanov":"Инженер_Алексей Рязанов",
        "bunin_arkady":"Инженер_Бунин Аркадий",
        "doppler_anton":"Инженер_Доплер Антон",
        "igor_eselevsky":"Инженер_Игорь Еселевский",
        "igor_lopatin":"Инженер_Игорь Лопатин",
        "kuzin_artem":"Инженер_Кузин Артем",
        "kucheryavykh_pavel":"Инженер_Кучерявых П.М.",
        "travin_evgeniy":"Инженер_Травин Евгений",
        "muhail_loginov":"Инженер_Михаил Логинов",
        "bainov_s.a.":"Инженер_Байнов С.А."}

    resultModelParams = {
        'Сопротивление изоляции: кол-во линий': 0,
        'Фаза-нуль: количество линий': 0,
        'Наличие цепи: кол-во точек':0,
        'Количество щитов':0,
        'Автоматы: Количество Автоматов':0,
        'УЗО: Количество УЗО':0,
        'Заземлителей: Кол-во ЗУ':0,
        'Молниезащита: Кол-во МЗ':0,
        'Электроустановка_Новая - Не удовлетворительно': 0,
        'Электроустановка_Новая - Удовлетворительно': 0,
        'Электроустановка_Старая - Не удовлетворительно': 0,
        'Электроустановка_Старая - Удовлетворительно': 0,
        'Цель испытаний_Приёмо-сдаточные': 0,
        'Цель испытаний_Эксплуатационные': 0,
        'Инженер_Аксенов Сергей': 0,
        'Инженер_Алексей Рязанов': 0,
        'Инженер_Бунин Аркадий': 0,
        'Инженер_Доплер Антон': 0,
        'Инженер_Игорь Еселевский': 0,
        'Инженер_Игорь Лопатин': 0,
        'Инженер_Кузин Артем': 0,
        'Инженер_Павел Кучерявых': 0,
        'Инженер_Травин Евгений': 0,
        'Инженер_Михаил Логинов': 0,
        'Инженер_Байнов С.А.':0}

    resultModelParams["Сопротивление изоляции: кол-во линий"] = params["resistanceNumberOfLines"]
    resultModelParams["Фаза-нуль: количество линий"] = params["phaseZeroNumberOfLines"]

    if params["settingNewness"] == 'new':
        if params["settingCondition"] == 'normal':
            resultModelParams["Электроустановка_Новая - Удовлетворительно"] = 1
        else:
            resultModelParams["Электроустановка_Новая - Не удовлетворительно"] = 1
    else:
        if params["settingCondition"] == 'normal':
            resultModelParams["Электроустановка_Старая - Удовлетворительно"] = 1
        else:
            resultModelParams["Электроустановка_Старая - Не удовлетворительно"] = 1


    if params["testTypes"] == 'operational':
        resultModelParams["Цель испытаний_Эксплуатационные"] = 1
    else:
        resultModelParams["Цель испытаний_Приёмо-сдаточные"] = 1

    resultModelParams[engineers[params["engineer"]]]=1

    return resultModelParams



@app.get('/')
def get_root():
    return {'message': 'the server is running'}

@app.get('/time/')
def predictor(resistanceNumberOfLines: int,  phaseZeroNumberOfLines: int, settingNewness: str, settingCondition: str,
              testTypes: str, engineer: str):
    listParams={
        "resistanceNumberOfLines": resistanceNumberOfLines,
        "phaseZeroNumberOfLines": phaseZeroNumberOfLines,
        "settingNewness": settingNewness,
        "settingCondition": settingCondition,
        "testTypes": testTypes,
        "engineer": engineer,
    }
    modelParams = pd.DataFrame(getParamsForModel(listParams), index=[0])
    joblib_LR_model = joblib.load("model.joblib")
    return joblib_LR_model.predict(modelParams).tolist()

list1 = '/reports-9.csv'




@app.get('/pretraining/')
def pretraining(list1):
    x, y = data.clearing_data(list1)
    to_model = model.model(x, y)
    return dump(to_model, '/model.joblib')

