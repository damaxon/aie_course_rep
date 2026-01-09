# HW06 – Report

## 1. Dataset

- Выбранный датасет: `S06-hw-dataset-04.csv`
- Размер: (5, 62)
- Целевая переменная: `target`, распределение "0"-0.9508,"1"-0.0492
- Признаки: числовые

## 2. Protocol

- Разбиение: train/test, random_state=42,stratify=y,размеры выборок:
  train: (18750, 60)
  test:  (6250, 60)

Распределение классов (train):
target
0    0.950773
1    0.049227
Name: proportion, dtype: float64

Распределение классов (test):
target
0    0.95088
1    0.04912
- Подбор: CV на train, 5 фолдов, оптимизировали параметр F1-score и гиперпараметр C
- Метрики: accuracy, F1, ROC-AUC, считаю уместным использование именно этих метрик в данной задаче, потому что accuracy нужен для доли верных предсказаний, а остальные две метрики важны тем что они устойчивы к дисбалансу классов, что в нашем случае будет главным фактором

## 3. Models

- DummyClassifier (baseline)
- LogisticRegression (baseline из S05)
- DecisionTreeClassifier (подбирали: `max_depth` + `min_samples_leaf` + `ccp_alpha`)
- RandomForestClassifier (подбирали: `max_depth` + `min_samples_leaf` + `max_features`)
- HistGradientBoosting (подбирали: `learning_rate` + `max_depth` + `max_leaf_nodes`)
- StackingClassifier (с CV-логикой)

## 4. Results

- Таблица финальных метрик на test по всем моделям
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>accuracy</th>
      <th>f1</th>
      <th>roc_auc</th>
      <th>model</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>0.98160</td>
      <td>0.776699</td>
      <td>0.906037</td>
      <td>Stacking</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.97984</td>
      <td>0.742857</td>
      <td>0.904646</td>
      <td>HistGradientBoosting</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.97024</td>
      <td>0.567442</td>
      <td>0.903547</td>
      <td>RandomForest</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.96256</td>
      <td>0.409091</td>
      <td>0.840041</td>
      <td>LogReg(scaled)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.96848</td>
      <td>0.588727</td>
      <td>0.827972</td>
      <td>DecisionTree</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.95088</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>Dummy(most_frequent)</td>
    </tr>
  </tbody>
</table>
</div>

- Лучшая модель вышла Stacking (по всем метрикам), так как с таким дисбалансом классов она показала лучшие результаты по метрикам, которые довольно устойчивы к дисбалансу

## 5. Analysis

- Устойчивость: при тестировании смены `random_state` (при этом оставляя stratify=y) на моделях LogReg и RandomForest, могу сделать вывод что результаты довольно стабильные, отклонения почти нет 
- Confusion matrix для лучшей модели (Stacking) показал:
TN: 5935 правильно предсказанных отрицательных
FP: всего 8 ложных срабатываний
FN: пропущено 107 положительных( главная ошибка )
TP: 200 правильно предсказанных положительных
- Интерпретация: permutation importance (top-10/15)  показал, что разница в важности 1 и последнего в top-15 признаков довольно большая, что говорит нам о том, что все дальшейшие признаки (от top-15 до top-60) скорее всего намного менее важны, чем первые в топе, поэтому могу предположить что довольно много признаков можно исключить почти без потери качества

## 6. Conclusion

- Деревья: быстро переобучаются на шумных данных, важна регуляризация и калибровка вероятностей

- Ансамбли: более устойчивы к шуму, но требуют большей настройки

- Стекинг: комбинация нескольких моделей дает хороший прирост к производительности

- Stratify + random_state: гарант воспроизводимости и одинакового распределения классов

- Интерпретируемость: permutation importance показывает важность признаков без зависимости от модели
