import tkinter as tk
from tkinter import ttk  # Импортируем ttk для создания Combobox
import pandas as pd
import sys
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Добавляем корневую директорию в системный путь
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

# Загрузка моделей
with open('models/catboost_model.pkl', 'rb') as f:
    model_catboost = pickle.load(f)

with open('models/xgboost_model.pkl', 'rb') as f:
    model_xgboost = pickle.load(f)

with open('models/randomforest_model.pkl', 'rb') as f:
    model_randomforest = pickle.load(f)

# Импортирование функций
from functions import predict_future_from_date, main_recommend_tender_volume_

# Загрузка обработанных тестовых данных
test_prepared = pd.read_excel('data/test_prepared.xlsx')

# Создание основного окна
root = tk.Tk()
root.title("Прогнозирование цен на арматуру")

# Ввод даты с выпадающим списком
tk.Label(root, text="Выберите дату:", font=("Times New Roman", 16)).pack()  # Увеличение размера шрифта
date_var = tk.StringVar()
unique_dates = test_prepared['dt'].dt.strftime('%Y-%m-%d').unique().tolist()
date_combobox = ttk.Combobox(root, textvariable=date_var, values=unique_dates)
date_combobox.pack()

# Выбор модели
tk.Label(root, text="Выберите модель:", font=("Times New Roman", 16)).pack()  # Увеличение размера шрифта
model_var = tk.StringVar(value='CatBoost')
tk.Radiobutton(root, text='CatBoost', variable=model_var, value='CatBoost', font=("Times New Roman", 14)).pack()
tk.Radiobutton(root, text='XGBoost', variable=model_var, value='XGBoost', font=("Times New Roman", 14)).pack()
tk.Radiobutton(root, text='RandomForest', variable=model_var, value='RandomForest', font=("Times New Roman", 14)).pack()

# Ввод количества недель
tk.Label(root, text="Введите количество недель (от 1 до 6):", font=("Times New Roman", 16)).pack()  # Увеличение размера шрифта
weeks_entry = tk.Entry(root, font=("Arial", 12))  # Увеличение размера шрифта
weeks_entry.pack()

# Метка для вывода результатов
result_label = tk.Label(root, text="", wraplength=500, justify="left", font=("Times New Roman", 16))  # Увеличение размера шрифта
result_label.pack()

# Функция для рисования графика
def draw_plot(predictions, model_name, start_date, n):
    plt.figure(figsize=(10, 6))
    plt.plot(test_prepared['dt'], test_prepared['Цена на арматуру'], label='Фактические значения', color='blue')

    future_dates = pd.date_range(start=start_date + pd.Timedelta(weeks=1), periods=n, freq='W-MON')
    
    plt.plot(future_dates, predictions, label='Предсказанные значения', color='red', linestyle='--')

    plt.title(f'Прогноз на {n} недель вперед by {model_name}')
    plt.xlabel('Дата')
    plt.ylabel('Цена на арматуру')
    
    # Вертикальная линия для выбранной даты
    plt.axvline(x=start_date, color='green', linestyle='--', label='Выбранная дата   ')
    plt.text(start_date, max(predictions), f'Выбранная дата: {start_date.date()}', color='green', ha='right')

    # Указываем расположение текста для количества недель
    # Можно разместить текст в конце графика, например, на максимальном значении по оси Y
    plt.text(future_dates[-1], max(predictions) + 2000, f'Количество недель: {n}', color='orange', ha='right', fontsize=12)

    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Функция для обработки ввода данных
def calculate_forecast():
    try:
        start_date = date_var.get()  # Получаем выбранную дату из Combobox
        model_name = model_var.get()
        weeks = int(weeks_entry.get())
        
        # Проверка на допустимый диапазон
        if weeks < 1 or weeks > 6:
            result_label.config(text="Ошибка: Введите количество недель от 1 до 6.")
            return  # Выход из функции, если значение некорректно
        
        # Преобразование даты
        start_date = pd.to_datetime(start_date)

        # Вызов функции предсказания
        if model_name == 'CatBoost':
            predictions = predict_future_from_date(model_catboost, start_date, weeks)
        elif model_name == 'XGBoost':
            predictions = predict_future_from_date(model_xgboost, start_date, weeks)
        elif model_name == 'RandomForest':
            predictions = predict_future_from_date(model_randomforest, start_date, weeks)

        # Вычисление объема тендера
        recommended_volume = main_recommend_tender_volume_(predictions, model_name)

        # Вывод результатов в метке
        result_label.config(text=f"Рекомендуемый объем тендера: {recommended_volume}")
        
        # Рисуем график
        draw_plot(predictions, model_name, start_date, weeks)
    
    except ValueError:
        result_label.config(text="Ошибка: Введите корректное количество недель.")
    except Exception as e:
        result_label.config(text=f"Ошибка: {str(e)}")

# Кнопка для расчета
tk.Button(root, text="Рассчитать прогноз", command=calculate_forecast).pack()

# Запуск интерфейса #####
root.mainloop()