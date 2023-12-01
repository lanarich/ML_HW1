# Машинное обучение. Домашнее задание 1
Выполнил: **Мустафин Фарид**

**Общий итог работы**   

В рамках домашнего задания была проделана следующая работа:
* Были обучены 6 моделей регрессии для предсказания стоимости автомобилей
* Реализован веб-сервис для применения построенной модели на новых данных

Если говорить подробнее о каждом пункте, то сначала мы получили "сырые" данные .csv. Был произведен базовый разведочный анализ, который показал, что данные содержат пропущенные значения, выбросы и плохо закодированные колонки. Мы исправили это заменив пропуски на медианные значения, удалили "экстримальные" выбросы за пределами 3 st.d и перекодировали колонки, почитсив их от единиц измерения. Далее с помощью графиков попытались предположить есть ли вообще связь между целевой переменной стоимости и предикторами, описали наблюдаемых характер связей. Немаловажно, что мы также рассмотрели связи между предкторами, чтобы проверить возможное наличие мультиколлинеарсности в данных. Между парой признаков к-т корреляции Пирсона = 0,68, что может намекать на мультиколлинеарность.  

Произведя предварительные манипуляции с данными мы попробовали построить базовую модель только с вещественными признаками и без регуляризации. Признаки предварительно были стандартизированы. *Модель 1* выдала следующий результат: $R^2$(трейн): 0.60, MSE(трейн): 113837878287.48, $R^2$(тест): 0.58, MSE(тест): 241836111915.74. Неплохо, но можно лучше, подумали мы. Попробовали Lasso-регрессию, *Модель 2* выдала следующий результат: R$R^2$(трейн): -23713.96, MSE(трейн): 6797616404885907.00, $R^2$(тест): -11852.20, MSE(тест): 6813556267082727.00. Явно получили плохую модель ($R^2$ - отрицательный), также Lasso-регрессия занулила коэффициенты для переменных: km_driven, engine, torque, max_torque_rpm. Это подсказывало нам, что возможно эти переменные незначимы. Мы попробовали сделать модель лучше с помощью перебора по сетке (c 10-ю фолдами, GridSearchCV). Получилась модель лушче, оптимальный параметр alpha = 10.0. *Модель 3* выдала : R^2(трейн): 0.60, MSE(трейн): 113837879649.65, R^2(тест): 0.58, MSE(тест): 241845450658.77. По качеству модель 3 такая же как и модель 1. Далее попробовали перебор по сетке (c 10-ю фолдами) для ElasticNet регрессии. Оптимальные параметры: alpha = 0.02, l1_ratio = 0.6. *Модель 4* выдала: R^2(трейн): 0.60, MSE(трейн): 113853709161.87, R^2(тест): 0.58, MSE(тест): 242885634289.06. В принципе ничего неожиданного.  

Далее мы решили добавить категориальные фичи. Порядковые переменные мы заранее отформатировали, а также перекодировали в дамми переменные. *Модель 5* Ridge-регрессии выдала: R^2(трейн): 0.67, MSE(трейн): 93542986939.98, R^2(тест): 0.35, MSE(тест): 372329347491.72. Заметно, что модель существенно ошибается на тесте. Чтобы это исправить был проведен Feature Engineering, в рамках которого добавились новые признаки, которые были получены путем объединения старых, а также была проведена более осмысленная обработка категориальных признаков: из них получалось много дамми переменных, которые ничего существенного не обозначали. У переменных owner и seats мы схлопнули некоторые категории, также именно на этом моменте мы убрали выбросы. Также мы завели последнюю *модель 6* Ridge-регрессии с преобразованными признаками и получили: R^2(трейн): 0.66, MSE(трейн): 97000202117.99, R^2(тест): 0.68, MSE(тест): 182780557274.72. Получили самую лучшую модель, которая не переобучилась и выдает достаточно хороший результат. Именно эта модель стала итоговой, которую мы и реализовали в веб-сервисе.  

В качестве дополнения мы сделали кастомную метрику -- среди всех предсказанных цен на авто посчитать долю предиктов, отличающихся от реальных цен на эти авто не более чем на 10%.

Веб-сервис был реализован с помощью Fast-api, для которого отдельно был создан код (файл main.py). Работает он достаточно просто и реализует 2 post запроса:
1. На вход в формате json подаются признаки одного объекта, на выходе сервис выдает предсказанную стоимость машины
2. На вход подается csv-файл с признаками тестовых объектов, на выходе получаем файл с +1 столбцом - предсказаниями на этих объектах

Вот как это выглядит:
![](https://clck.ru/36tRKN)
