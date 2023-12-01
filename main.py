#Все по классике, сначала грузим пакеты, потом делаем app

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
import pickle
import pandas as pd
import sklearn
from fastapi.responses import FileResponse
import io


app = FastAPI()

#Описание данных для одиночного наблюдения

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

#Загрузка модели, которую мы обучили на наших тренировочных данных

def load_model():
    with open('ridge_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

#Предобработка данных. Так как самая лучшая модель - с категориальными переменными, значит поступающие данные также
#необходимо обработать, более подробно об обработке признаков в ноутбуке

def preprocess_data(item: Item):
    #для того, чтобы проверить структуру данных, обработанных и тех, что потребляет модель, чтобы было гладко
    df = {'year': 2014.0,
    'km_driven': 145500.0,
    'mileage': 23.4,
    'engine': 1248.0,
    'max_power': 74.0,
    'torque': 190.0,
    'max_torque_rpm': 2000.0,
    'bph/CC': 0.05929487179487179,
    'year2': 4056196.0,
    'fuel_Diesel': 1.0,
    'fuel_LPG': 0.0,
    'fuel_Petrol': 0.0,
    'seller_type_Individual': 1.0,
    'seller_type_Trustmark Dealer': 0.0,
    'transmission_Manual': 1.0,
    'owner_3+(test)': 0.0,
    'seats_6-7': 0.0,
    'seats_8+': 0.0}

    df = pd.DataFrame(dict(df), index=[0])
    data = pd.DataFrame(dict(item), index=[0])


    columns = ['mileage', 'engine', 'max_power']
    for col in columns:
        data[col] = pd.to_numeric(data[col].str.extract('(\d+(\.\d+)?)', expand=False)[0], errors='coerce')


    data['max_torque_rpm'] = pd.to_numeric(
        data['torque'].str.extract(r'([\d,.]+)(?=\D*$|$)')[0].str.replace(',', ''), errors='coerce')
    c = data['torque'].str.extract(r'(^[\d.]+)').astype('float')
    b = data['torque'].str.contains('Nm', case=False)
    data['torque'] = c[0] * b.map({True: 1, False: 9.80665})

    data['bph/CC'] = data['max_power'] / data['engine']
    data['year2'] = data['year'] ** 2

    bins = [2, 6, 8, 20]
    data['seats'] = pd.cut(data['seats'], bins=bins, labels=['2-5', '6-7', '8+'], right=False)
    data['owner'] = data['owner'].replace(
        {'First Owner': '1-2', 'Second Owner': '1-2', 'Third Owner': '3+(test)', 'Fourth & Above Owner': '3+(test)',
         'Test Drive Car': '3+(test)'})
    data = data.drop(['selling_price', 'name'], axis=1)
    data = pd.get_dummies(data, columns=['fuel', 'seller_type', 'transmission', 'owner', 'seats'], drop_first=True)

    #вот тут как раз и дополняем колонки того чего не было при создании дамми переменных.
    data = data.reindex(columns=df.columns, fill_value=0)
    return data

def preprocess_сsv(dataframe):
    #Тоже самое, но уже для датафрейма, тут нет перевода словаря в датафрей, сразу работа с файлом
    df = {'year': 2014.0,
    'km_driven': 145500.0,
    'mileage': 23.4,
    'engine': 1248.0,
    'max_power': 74.0,
    'torque': 190.0,
    'max_torque_rpm': 2000.0,
    'bph/CC': 0.05929487179487179,
    'year2': 4056196.0,
    'fuel_Diesel': 1.0,
    'fuel_LPG': 0.0,
    'fuel_Petrol': 0.0,
    'seller_type_Individual': 1.0,
    'seller_type_Trustmark Dealer': 0.0,
    'transmission_Manual': 1.0,
    'owner_3+(test)': 0.0,
    'seats_6-7': 0.0,
    'seats_8+': 0.0}

    df = pd.DataFrame(dict(df), index=[0])


    columns = ['mileage', 'engine', 'max_power']
    for col in columns:
        dataframe[col] = pd.to_numeric(dataframe[col].str.extract('(\d+(\.\d+)?)', expand=False)[0], errors='coerce')


    dataframe['max_torque_rpm'] = pd.to_numeric(
        dataframe['torque'].str.extract(r'([\d,.]+)(?=\D*$|$)')[0].str.replace(',', ''), errors='coerce')
    c = dataframe['torque'].str.extract(r'(^[\d.]+)').astype('float')
    b = dataframe['torque'].str.contains('Nm', case=False)
    dataframe['torque'] = c[0] * b.map({True: 1, False: 9.80665})

    dataframe['bph/CC'] = dataframe['max_power'] / dataframe['engine']
    dataframe['year2'] = dataframe['year'] ** 2

    bins = [2, 6, 8, 20]
    dataframe['seats'] = pd.cut(dataframe['seats'], bins=bins, labels=['2-5', '6-7', '8+'], right=False)
    dataframe['owner'] = dataframe['owner'].replace(
        {'First Owner': '1-2', 'Second Owner': '1-2', 'Third Owner': '3+(test)', 'Fourth & Above Owner': '3+(test)',
         'Test Drive Car': '3+(test)'})
    dataframe = dataframe.drop(['selling_price', 'name'], axis=1)
    dataframe = pd.get_dummies(dataframe, columns=['fuel', 'seller_type', 'transmission', 'owner', 'seats'], drop_first=True)
    dataframe = dataframe.reindex(columns=df.columns, fill_value=0)
    return dataframe


#Запрос на предсказание одиночного наблюдения. Также внутри загрузка модели и обработка данных. Выводим предикт
@app.post("/predict_item")
def predict_item(item: Item):
    trained_model = load_model()
    processed_data = preprocess_data(item)
    prediction = trained_model.predict(processed_data)

    return {"prediction": prediction[0]}

#Запрос на добавление предсказанных значений в файл. Внутри: получаем файл в битах, переделываем в датафрей.
#Потом предсказываем и склеиваем первоначальный df_old с предсказаниями. Отдаем полученный файл.
@app.post("/predict_items")
def csv_predictions(file: UploadFile = File(...)):
    contents = file.file.read()
    df = pd.read_csv(io.BytesIO(contents))
    df_old = df.copy()
    trained_model = load_model()
    processed_data = preprocess_сsv(df)
    prediction = trained_model.predict(processed_data)
    df_final = pd.concat([df_old, pd.DataFrame(prediction, columns=['selling_price_pred'])], axis = 1)
    df_final.to_csv("predictions.csv", index=False)
    return FileResponse("predictions.csv", media_type="text/csv", filename="predictions.csv")


