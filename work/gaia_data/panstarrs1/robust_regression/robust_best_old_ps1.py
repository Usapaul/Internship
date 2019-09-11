#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import least_squares

import matplotlib.pyplot as plt
from matplotlib import rcParams
from astropy.io import ascii

import statsmodels.api as sm
import statsmodels.formula.api as smf

#------------------------------------------------

filename = 'for_robust_mag_err_0p02_260k_noBLUEEE_Xn1_visP8.dat'
tbl = ascii.read(filename)

col_g = 'phot_g_mean_mag'
col_bp = 'phot_bp_mean_mag'
col_rp = 'phot_rp_mean_mag'
col_bp_rp = 'bp_rp'
col_bp_g = 'bp_g'
col_g_rp = 'g_rp'

col_g_ps1 = 'g_mean_psf_mag'
col_r_ps1 = 'r_mean_psf_mag'
col_i_ps1 = 'i_mean_psf_mag'
col_z_ps1 = 'z_mean_psf_mag'
col_y_ps1 = 'y_mean_psf_mag'


convert_to_numpy_array = ascii.convert_numpy(np.float)[0]

# Если хотим использовать центрированные данные,
# то centering = True 
centering = False

def create_y(cols, centered=True):
	# В списке cols может быть либо один column, 
	# либо два. И надо это будет проверять по его длине
	# и затем будет сделан выбор: работаем только со
	# значениями в одном фильтре, или с разностью в двух 
	col_1 = cols[0]
	filter1 = convert_to_numpy_array(tbl[col_1])
	if len(cols) > 1:
		col_2 = cols[1]
		filter2 = convert_to_numpy_array(tbl[col_2])
		y = filter1 - filter2
	else:
		y = filter1
	if centered == True:
		median_y = np.median(y)
		y -= median_y
		global m_y
		m_y = median_y
	return y

def create_var1(col,centered=True):
	x1 = convert_to_numpy_array(tbl[col])
	if centered == True:
		median_x1 = np.median(x1)
		x1 -= median_x1
		global m_x1
		m_x1 = median_x1
	return x1

def create_var2(col,centered=True):
	x2 = convert_to_numpy_array(tbl[col])
	if centered == True:
		median_x2 = np.median(x2)
		x2 -= median_x2
		global m_x2
		m_x2 = median_x2
	return x2


# m_y, m_x1 и m_x2 будут хранить значение медианы массивов 
# y и переменных x1 и x2 соответственно. Для начала просто
# создаются эти переменные, и потом, если 
# centering == True, они получат ненулевые значения
m_y = 0.
m_x1 =0.
m_x2 = 0.
#------------------------------------------------

# В x_cols будут храниться имена переменных -- значений в 
# фильтрах или цвет. И тогда в x_degs должно быть указано
# такое же количество переменных типа integer, которые 
# указывают степень полинома от соответствующей переменной 
# Степень x1 не больше двух, степень x2 не больше трех

# В поле, ограниченное ">>> >>>", нужно вводить данные

# ЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖ
# ЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖ
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# DATA HERE DATA HERE DATA HERE DATA HERE DATA
# >>>
y_cols = [col_i_ps1]
x_cols = [col_rp,col_bp_rp]
# x_cols = [col_rp,col_bp_rp]
x_degs = [1,3]
# x_degs = [1,3]
# >>>
# DATA HERE DATA HERE DATA HERE DATA HERE DATA
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# ЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖ
# ЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖ


if len(x_cols) != len(x_degs):
	raise SystemExit('STOP: len(x_cols) /= len(x_degs)')

y_array = create_y(y_cols, centering)

# Переменных x может быть одна или две, и они должны быть записаны
# столбцами в соответствующих степенях.
# Следующий набор команд это и делает:
x_var1 = create_var1(x_cols[0], centering)
x_array = x_var1
for i in range(2,x_degs[0]+1):
	x_array = np.column_stack((x_array,x_var1**i))
if len(x_cols) > 1:
	x_var2 = create_var2(x_cols[1], centering)
	x_array = np.column_stack((x_array,x_var2))
	for i in range(2,x_degs[1]+1):
		x_array = np.column_stack((x_array,x_var2**i))		
x_array = sm.add_constant(x_array, prepend=True)


rlm_model = sm.RLM(y_array, x_array, M=sm.robust.norms.TrimmedMean())
results = rlm_model.fit()

# -----------------------------------------------

def centering_back(params,degs):
	A, B1, C1, B2, C2, D2 = 0., 0., 0., 0., 0., 0. 
	A = params[0]
	if len(degs) == 1:
		global m_x1, m_x2
		# Потому что при построении x1 было получено m_x1, а 
		# m_x2 так и осталось нулем. Но тут я работаю с одной
		# переменной как со второй (это подтверждают двойки 
		# в названии используемых в этом блоке if переменных
		m_x1, m_x2 = m_x2, m_x1
		B2 = params[1]
		if degs[0] > 1:
			C2 = params[2]
		if degs[0] > 2:
			D2 = params[3]
	if len(degs) == 2:
		B1 = params[1]
		if degs[0]>1:
			C1 = params[2]
			B2 = params[3]
			if degs[1] > 1:
				C2 = params[4]
			if degs[1] > 2:
				D2 = params[5]
		else:
			B2 = params[2]
			if degs[1] > 1:
				C2 = params[3]
			if degs[1] > 2:
				D2 = params[4]
	a = m_y + A - B1*m_x1 + C1*m_x1**2 - B2*m_x2 + C2*m_x2**2 - D2*m_x2**3
	b1 = B1 - 2*C1*m_x1
	c1 = C1
	b2 = B2 - 2*C2*m_x2 + 3*D2*m_x2**2
	c2 = C2 - 3*D2*m_x2
	d2 = D2
	if len(degs) == 1:
		if degs[0] == 1:
			return np.array([a,b2])
		elif degs[0] == 2:
			return np.array([a,b2,c2])
		else:
			return np.array([a,b2,c2,d2])
	else:
		if degs[0] == 1:
			if degs[1] == 1:
				return np.array([a,b1,b2])
			elif degs[1] == 2:
				return np.array([a,b1,b2,c2])
			else:
				return np.array([a,b1,b2,c2,d2])
		else:
			if degs[1] == 1:
				return np.array([a,b1,c1,b2])
			elif degs[1] == 2:
				return np.array([a,b1,c1,b2,c2])
			else:
				return np.array([a,b1,c1,b2,c2,d2])


def mdisp(X,W):
    M = np.sum(W*X)/np.sum(W)
    sW = np.sum(W)
    V = np.sum(W*(X-M)**2)*sW/(sW**2-np.sum(W**2))
    S = np.sqrt(V)
    return S

# -----------------------------------------------

print('------------------------------------------')
print(filename)
print('TukeyBiweight')
print('y:      '+str(y_cols))
print('x:      '+str(x_cols))
print('x_degs: '+str(x_degs))
print('')

print('Parameters:')
if centering == True:
	parameters = centering_back(results.params,x_degs)
	print(parameters)
	print('')
	print('Centering ON')
	print('Parameters and std err, centered data:')
	print(results.params)
else:
	parameters = results.params
	print(parameters)
	print('')
	print('Std err:')

print(results.bse)
print('')
print('dispersion:')
print(mdisp(results.resid,results.weights))
print('')

pvalues = results.pvalues
for i in range(len(pvalues)):
	pvalues[i] = round(pvalues[i],3)
print('Pvalues:')
print(pvalues)




