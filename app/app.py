import tkinter as tk
from tkinter import messagebox
import pandas as pd
import sys
import os
import pickle

# Добавляем корневую директорию в системный путь
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

# Загрузка моделей
with open('models/catboost_model.pkl', 'rb') as f:
    model_catboost = pickle.load(f)

with open('models/xgboost_model.pkl', 'rb') as f:
    model_xgboost = pickle.load(f)

with open('models/randomforest_model.pkl', 'rb') as f:
    model_randomforest = pickle.load(f)

# Теперь можно импортировать функции из main.ipynb
from functions import predict_future_from_date, main_recommend_tender_volume_

# Функция для обработки ввода данных
def calculate_forecast():
    try:
        start_date = date_entry.get()
        model_name = model_var.get()
        weeks = int(weeks_entry.get())
        
        # Преобразование даты
        start_date = pd.to_datetime(start_date)

        # Вызов функции предсказания
        if model_name == 'CatBoost':
            predictions = predict_future_from_date(model_catboost, start_date, weeks)
        elif model_name == 'XGBoost':
            predictions = predict_future_from_date(model_xgboost, start_date, weeks)
        elif model_name == 'RandomForest':
            predictions = predict_future_from_date(model_randomforest, start_date, weeks)

        # Вывод результатов
        messagebox.showinfo("Результаты", f"Прогноз на {weeks} недель вперед by {model_name}: {predictions}")
    
    except Exception as e:
        messagebox.showerror("Ошибка", str(e))

# Создание основного окна
root = tk.Tk()
root.title("Прогнозирование цен на арматуру")

# Ввод даты
tk.Label(root, text="Введите дату (YYYY-MM-DD):").pack()
date_entry = tk.Entry(root)
date_entry.pack()

# Выбор модели
tk.Label(root, text="Выберите модель:").pack()
model_var = tk.StringVar(value='CatBoost')
tk.Radiobutton(root, text='CatBoost', variable=model_var, value='CatBoost').pack()
tk.Radiobutton(root, text='XGBoost', variable=model_var, value='XGBoost').pack()
tk.Radiobutton(root, text='RandomForest', variable=model_var, value='RandomForest').pack()

# Ввод количества недель
tk.Label(root, text="Введите количество недель:").pack()
weeks_entry = tk.Entry(root)
weeks_entry.pack()

# Кнопка для расчета
tk.Button(root, text="Рассчитать прогноз", command=calculate_forecast).pack()

# Запуск интерфейса
root.mainloop()