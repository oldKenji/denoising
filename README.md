# Инструкция для использования модели.

Модель основана на архитектуре [Conv-TasNet](https://arxiv.org/pdf/1809.07454.pdf).

Модель позволяет проводить процедуру denoising mel-спектрограмм в формате файлов numpy.  
Одновременно, есть возможность проверки mel-спектрограмм на наличие зашумления.

## Структура проекта:
***data*** - предложенный каталог для хранения данных с примером тестовой выборки.

***input*** - предложенный каталог для хранения одиночных mel-спектрограмм с примером.

***output*** - каталог с выходными обработанными одиночными mel-спектрограммами с примером.

***model.py*** - модель.

***dataset.py*** - датасеты для модели.

***train.py*** - тренировка модели.

***denoising.py*** - проверка работы модели в режиме **denoising** и обработка одиночных mel-спектрограмм.

***detect.py*** - проверка работы модели в режиме **classification** и обработка одиночных mel-спектрограмм.

***utils_gz.py*** - дополнительные используемые функции.

***README.md*** - данная инструкция по использованию.

***model.pth*** - предобученные параметры модели.

## Для проверки и обработки (denoising)

Для проверки работы модели в режиме denoising используется следующий формат записи:
> py denoising.py -i data -m test

где:  
**data** - путь к проверочному датасету.  
**test** - название каталога проверочного датасета.

Для данной записи соответствует следующая структура каталогов:  
**data**  
-**test**  
--clean  
--noisy  

Результатом выполнения будет вывод значения MSE для тестовой выборки.

Для обработки одиночных mel-спектрограмм используется следующий формат записи:
> py denoising.py -i input\mel.npy  

где:  
**input\mel.npy** - путь к обрабатываемому файлу.

Результат обработки сохранится в папке **output** под именем **cleaned_mel.npy**.  

## Для обучения (denoising)

Для обучения модели используется следующий формат записи:
> py train.py

Возможен запуск на предложенных данных для проверки(на CPU займет несколько минут).  
Для запуска полноценного обучения, требуется скопировать **train** и **val** выборки в папку **data** с сохранением 
структуры каталогов.

## Для проверки и обработки (classification)

Для проверки работы модели в режиме classification используется следующий формат записи:
> py detect.py -i data -m test

где:  
**data** - путь к проверочному датасету.  
**test** - название каталога проверочного датасета.

Для данной записи соответствует следующая структура каталогов:  
**data**  
-**test**  
--clean  
--noisy  

Результатом выполнения будет вывод значения ACCURACY для тестовой выборки.

Для обработки одиночных mel-спектрограмм используется следующий формат записи:
> py detect.py -i input\mel.npy  

где:  
**input\mel.npy** - путь к обрабатываемому файлу.

## Дополнительная информация

Для обучения модели использовалась функция потерь MSE.  
Если бы стояла задача создания __рабочей__ модели, я использовал бы более сложную функцию потерь, 
другую архитектуру и не использовал mel-спектрограммы.
