import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Загрузка обработанных тестовых данных
test_prepared = pd.read_excel('data/test_prepared.xlsx')

features_ = ['trend', 'seasonal', 'year', 'month', 'weekday', 'dayofyear', 'quarter', 
            'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
            'rolling_mean_3', 'rolling_mean_6', 
            'diff_1', 'diff_2', 'diff_3']

def predict_future_from_date(model, start_date, N):
    # Найти строку с данными для введенной даты
    row_data = test_prepared[test_prepared['dt'] == start_date]
    
    if row_data.empty:
        raise ValueError("Дата не найдена в тестовом наборе.")
    
    last_row = row_data[features_].values.reshape(1, -1)
    future_predictions = []

    for i in range(N):
        pred = model.predict(last_row)
        future_predictions.append(int(np.round(pred[0])))
        
        last_row = last_row.copy()
        # Обновляем месяц и год
        if last_row[0, features_.index('month')] < 12:  # Исправлено на 12, так как месяцев 12
            last_row[0, features_.index('month')] += 1  
        else:
            last_row[0, features_.index('month')] = 1  
            last_row[0, features_.index('year')] += 1 

        # Обновляем лаги
        last_row[0, features_.index('lag_5')] = last_row[0, features_.index('lag_4')]
        last_row[0, features_.index('lag_4')] = last_row[0, features_.index('lag_3')]
        last_row[0, features_.index('lag_3')] = last_row[0, features_.index('lag_2')]
        last_row[0, features_.index('lag_2')] = last_row[0, features_.index('lag_1')]
        last_row[0, features_.index('lag_1')] = pred  # Обновляем lag_1 на предсказанное значение

        # Обновление скользящих средних
        if i == 0:
            last_row[0, features_.index('rolling_mean_3')] = (last_row[0, features_.index('lag_1')] + last_row[0, features_.index('lag_2')]) / 2
            last_row[0, features_.index('rolling_mean_6')] = (last_row[0, features_.index('lag_1')] + last_row[0, features_.index('lag_2')] + last_row[0, features_.index('lag_3')]) / 3
        else:
            last_row[0, features_.index('rolling_mean_3')] = (last_row[0, features_.index('lag_1')] + last_row[0, features_.index('lag_2')] + future_predictions[-1]) / 3
            last_row[0, features_.index('rolling_mean_6')] = (last_row[0, features_.index('lag_1')] + last_row[0, features_.index('lag_2')] + future_predictions[-1]) / 3

    return future_predictions  # Возврат предсказанных значений

def visual(future_predict, model_name, n):
    plt.figure(figsize=(10, 6))
    plt.plot(test_prepared['dt'], test_prepared['Цена на арматуру'], label='Фактические значения', color='blue')

    future_dates = pd.date_range(start=test_prepared['dt'].iloc[-1] + pd.Timedelta(weeks=1), periods=n, freq='W-MON')

    # Убедимся, что future_predict имеет нужный размер
    if len(future_predict) > n:
        future_predict = future_predict[:n]  # Обрезаем до нужного размера

    plt.plot(future_dates, future_predict, label='Предсказанные значения', color='red', linestyle='--')

    plt.title(f'Прогноз на {n} недель вперед by {model_name}')
    plt.xlabel('Дата')
    plt.ylabel('Цена на арматуру')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Функция для рекомендации объема тендера
def main_recommend_tender_volume_(predictions, model_name):
    def perenos_():
        print('\n******************************************************************\n')

    def recommend_tender_volume(predicted_price, current_price, threshold_down=0.95, threshold_up=1.05, min_tender_weeks=1, max_tender_weeks=6, price_difference_factor=1000):
        if predicted_price < current_price * threshold_down:
            return max(min_tender_weeks, int((current_price - predicted_price) / price_difference_factor))
        elif predicted_price > current_price * threshold_up:
            return max(max_tender_weeks, int((predicted_price - current_price) / price_difference_factor))
        else:
            return min_tender_weeks + (max_tender_weeks - min_tender_weeks) // 2  # Возможно, нужно добавить случайное значение


    current_price = test_prepared['Цена на арматуру'].iloc[-1]  # цена на арматуру сегодня

    recommended_volumes = []

    # Вычисление тендера на основе предсказанных значений
    for predicted_price in predictions:
        recommended_volume = recommend_tender_volume(predicted_price, current_price)
        recommended_volumes.append(recommended_volume)
        print(f"Предсказанная цена: {predicted_price}, Рекомендуемый объем: {recommended_volume}")  # Отладочная информация

    # Проверка, есть ли рекомендованные объемы
    if not recommended_volumes:
        print("Не удалось рассчитать объем тендера: рекомендованные объемы пусты.")
        return None  # Возвращаем None, если массив пустой

    # Возврат среднего рекомендованного объема
    average_volume = int(np.mean(recommended_volumes))  # Возвращаем одно целое число
    print(f"Средний рекомендованный объем: {average_volume}")  # Отладочная информация
    return average_volume