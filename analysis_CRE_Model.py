#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 14:00:36 2024

@author: cliffng
"""

import numpy as np
import pandas as pd
import linearmodels as plm
from scipy import stats

# Import Data
nba = pd.read_csv('./data/finaldf.csv')
nba = nba.dropna().drop(columns = ['Coach', 'Perc_2PA'])

# Create group specific means
groupby_team = nba.drop(columns = ['Season', 'W']).groupby('TEAM').mean().add_suffix('_mean').reset_index()
nba = pd.merge(nba, groupby_team, on = 'TEAM', how = 'left')

# n, N and T
n = len(nba.TEAM.unique())
N = len(nba)
T = len(nba.Season.unique())

# Set Index
nba = nba.set_index(['TEAM', 'Season'])

# Selecting Variables
y = nba.W
X = nba.drop(columns = ['W'])

# Running CRE
reg_cre = plm.RandomEffects(y, X).fit(cov_type = "clustered")
print(reg_cre.summary)

k = X.shape[1] - 1
urss = reg_cre.resid_ss

# Wald Test for CRE
formula = ''
for i in list(groupby_team.columns):
    formula += f'{i} = '
formula = formula[7:]
formula += '0'
wtest = reg_cre.wald_test(formula = formula)
print(f'wtest: \n{wtest}\n')

# F-Test for CRE
nba = pd.read_csv('./data/finaldf.csv')
nba = nba.dropna().drop(columns = ["Perc_2PA", 'Coach'])

# n, N and T
n = len(nba.TEAM.unique())
N = len(nba)
T = len(nba.Season.unique())

nba = nba.set_index(['TEAM', 'Season'])

y = nba.W
X = nba.drop(columns = ['W'])

reg_re = plm.RandomEffects(y, X).fit(cov_type = "clustered")
print(reg_re)

k = X.shape[1] - 1
rrss = reg_re.resid_ss

ftest1 = ((rrss-urss)/(reg_re.df_resid - reg_cre.df_resid))/(urss/reg_cre.df_resid)
pval1 = 1-stats.f.cdf(abs(ftest1), reg_re.df_resid - reg_cre.df_resid, reg_cre.df_resid)
print('\nCRE F-Test')
print(f'F_Test: {round(ftest1, 4)}')
print(f'p-value: {round(pval1, 4)}')
print(f'Distribution: F({reg_re.df_resid - reg_cre.df_resid}, {reg_cre.df_resid})\n')