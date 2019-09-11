#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    Программа получает на вход файл с таблицей звезд и
    на выходе выдает файл с таблицей значений поглощения

    Запуск программы: python3 bayestarQuery.py [1] [2] [3] [4:]
    [1] -- имя файла с таблицей со столбцами 'id', 'l', 'b', 'd'
           где id -- номер звезды, l - галактическая долгота,
           b -- галактическая широта, d -- расстояние до звезды
    [2] -- название версии карты поглощения bayestar20{15,17,19}
    [3] -- тип метода запроса (mode) (mean, median, samples,...)
    [4:] -- значения перцентилей
'''


import numpy as np

from astropy.coordinates import SkyCoord
import astropy.units as units
from dustmaps.bayestar import BayestarWebQuery

from astropy.io import ascii
from astropy.table import Table

import sys

#------------------------------------------------


if len(sys.argv) > 2:
    bayestar_version = sys.argv[2]
else:
    bayestar_version = 'bayestar2019'
# bversion -- сокращенная запись версии, типа b15/b17/b19
bversion = 'b'+bayestar_version[-2:]
bayestar = BayestarWebQuery(version=bayestar_version)


# В data_file должны быть 4 столбца:
# id, l, b, d,
# где id -- номер звезды, l - галактическая долгота,
# b -- галактическая широта, d -- расстояние до звезды
if len(sys.argv) > 1:
    datafileName = sys.argv[1]
else:
    datafileName = input('Enter the name of data file: ')


# нужно выбрать один из типов mode, выдаваемых Bayestar:
# ‘random_sample’, ‘random_sample_per_pix’ ‘samples’,
# ‘median’, ‘mean’, ‘best’ или ‘percentile’ + указание pct. 
# mode определяет,как выдаваемые значения поглощения будут отражать 
# вероятностную природу 3D-карты поглощения
if len(sys.argv) > 3:
    bayestarMode = sys.argv[3]
else:
    bayestarMode = 'best'

if bayestarMode == 'percentile':
    if len(sys.argv) > 4:
        pcts = list(map(float,sys.argv[4:]))
    else:
        pcts = [16.,50.,84.]

#------------------------------------------------


print()
print('Data file:      ', datafileName)
print('Version of map: ', bayestar_version)
if bayestarMode == 'percentile':
    print('Mode type:      ', bayestarMode, pcts)
else:
    print('Mode type:      ', bayestarMode)

# ===============================================
# ===============================================


tbl_IDlbd = ascii.read(datafileName)
col_names = list(tbl_IDlbd.columns)
columns = col_id, col_l, col_b, col_d = 'id', 'l', 'b', 'd'
for i in range(len(columns)):
    tbl_IDlbd.rename_column(col_names[i],columns[i])


num_rows_data = len(tbl_IDlbd)

# numExtValues -- количество выдаваемых значений поглощения (может быть одно,
# а может быть несколько, если выбрать bayestarMode='samples')
# здесь coords_example используется только для того, чтобы получить
# значение numExtValues
if bayestarMode not in ('samples','percentile'):
    numExtValues = 1
    # extArray -- в нем будут храниться только то (значения поглощения), 
    # что непосредственно выдается по запросу BayestarQuery
    extArray = np.zeros((num_rows_data))
else:
    # отправляется тестовый запрос для определения numExtValues
    coords_example = SkyCoord(1*units.deg, 1*units.deg,distance=1*units.pc)
    if bayestarMode == 'percentile':
        numExtValues = len(bayestar(coords_example,mode=bayestarMode,pct=pcts))
    else:
        numExtValues = len(bayestar(coords_example,mode=bayestarMode))
    extArray = np.zeros((num_rows_data,numExtValues))


# выходной файл со значениями поглощения для каждой звезды
# называется как исходный, но с суффиксом _b[version]
outfileNname = datafileName.split('.')[0]+'_'+bversion+'_'+bayestarMode+'.dat'

# Bayestar Web Query имеет ограничение -- не более 100k строк за раз,
# поэтому общий запрос разбивается на num_sets запросов по 100k
# ограничение записывается в maxrows (вдруг когда-то изменится)
maxrows = 100000
num_sets = (num_rows_data - 1) // maxrows + 1

# num1Row -- номер первой считываемой строки data_file
# во время одного Query включительно
for numset in range(num_sets):
    num1Row = numset*maxrows 
    numrows = maxrows
    if numset == num_sets - 1:
        # если это последний запрос, то в нем может уже быть
        # меньше maxrows строк, так как это остаток
        # от деления числа строк в data_file на 100000
        numrows = (num_rows_data - 1) % maxrows + 1
    l = tbl_IDlbd[col_l][num1Row:num1Row+numrows] * units.deg
    b = tbl_IDlbd[col_b][num1Row:num1Row+numrows] * units.deg
    d = tbl_IDlbd[col_d][num1Row:num1Row+numrows] * units.pc
    coords = SkyCoord(l, b, distance=d, frame='galactic')
    if bayestarMode != 'percentile':
        extArray[num1Row:num1Row+numrows] = bayestar(coords,mode=bayestarMode)
    else:
        extArray[num1Row:num1Row+numrows] = bayestar(coords,mode=bayestarMode,
                                                     pct=pcts)
    print(round(num1Row/num_rows_data*100),'%',sep='')

if bayestarMode not in ('samples','percentile'):
    resultTable = Table([tbl_IDlbd[col_id],extArray],
                        names=('source_id',bversion+bayestarMode),
                        dtype=('i8','f2'))
else:
    resultTable = Table([tbl_IDlbd[col_id]],
                        names=(['source_id']),dtype=(['i8']))
    # Приходится долго возиться с совмещением таблицы с другой, в которой
    # несколько столбцов, да так, чтобы я дал имена столбцов правильно в 
    # цикле for -- оказалось трудно
    if bayestarMode == 'samples':
        col_names = [bversion+'_'+str(i) for i in range(1,numExtValues+1)]
    else:
        col_names = [bversion+'_pct'+str(round(pcts[i])) 
                     for i in range(numExtValues)]
    extTable = Table(extArray,names=col_names,
                     dtype=['f2' for i in range(numExtValues)])

    for i in range(numExtValues):
        # в resultTable добавляем по одному столбцу из extTable с col_names
        resultTable[col_names[i]] = extTable[col_names[i]]


ascii.write(resultTable,outfileNname,format='commented_header',overwrite=True)

print()
print('Result extinction table saved as '+outfileNname)
print()
