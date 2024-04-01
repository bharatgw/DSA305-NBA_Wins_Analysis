import pandas as pd 
import numpy as np
import scipy.stats as stats
import linearmodels as plm 
from linearmodels.panel import PooledOLS
import matplotlib.pyplot as plt
import linearmodels as lm
from scipy.stats import f, chi2
from scipy.optimize import minimize
from Panel2RE_MLE import ML_RandomEffect


nba = pd.read_csv("finaldf.csv")
nba = nba.dropna()

time = pd.Categorical(nba.Season)
indi = pd.Categorical(nba.TEAM)
nba = nba.set_index(["TEAM", "Season"])
nba["time"] = time
nba["indi"] = indi
nba = nba.drop(columns = ['Coach'])

# Test functions
def f_test(resMod, unresMod): 
    rss = resMod.resid_ss 
    urss = unresMod.resid_ss 
    r_df = resMod.df_resid 
    ur_df = unresMod.df_resid 
    diff_df = r_df - ur_df
    f_stat = ((rss - urss)/diff_df)/(urss/ur_df)
    print(f"(a): F-stat: {round(f_stat, 2)},\n p-value: {round(1-stats.f.cdf(f_stat,diff_df, ur_df), 2)},\n Distribution: F({diff_df}, {ur_df})")


#Pooled OLS model 
regMod = lm.PanelOLS(nba.dropna()["W"], nba.dropna().drop(["W", 'time', 'indi'], axis = 1)).fit(cov_type = "robust")
print(regMod.summary)

# One way fixed effect model 
onefe_reg = plm.PanelOLS.from_formula(formula = 'W ~ Perc_3PA + Perc_2PA + Perc_AST + Perc_STL +'
                                    'PFminusPFD + OPP_Perc_3PA + OPP_Perc_AST + OPP_Perc_STL +'
                                    'L1_N_Awards_Won + L1_Coach_RS_W_Perc_Overall + L1_Coach_P_W_Perc +'
                                    'AVG_PLAYER_AGE + Coach_N_Seasons_Overall + Coach_Perc_Seasons_TEAM +'
                                    'EntityEffects', data = nba, drop_absorbed = True) 
onefe_results = onefe_reg.fit(cov_type = "clustered")
print(onefe_results.summary)



# One way fixed effect model 
onefe_reg1 = plm.PanelOLS.from_formula(formula = 'W ~ Perc_3PA + Perc_2PA + Perc_AST + Perc_STL +'
                                    'PFminusPFD + OPP_Perc_3PA + OPP_Perc_AST + OPP_Perc_STL +'
                                    'L1_N_Awards_Won + L1_Coach_RS_W_Perc_Overall + L1_Coach_P_W_Perc +'
                                    'AVG_PLAYER_AGE + Coach_N_Seasons_Overall + Coach_Perc_Seasons_TEAM +'
                                    'TimeEffects', data = nba, drop_absorbed = True) 
onefe_results1 = onefe_reg1.fit(cov_type = "clustered")
print(onefe_results1.summary)       

# Two way fixed effect model
twofe_reg = plm.PanelOLS.from_formula(formula = 'W ~ Perc_3PA + Perc_2PA + Perc_AST + Perc_STL +'
                                    'PFminusPFD + OPP_Perc_3PA + OPP_Perc_AST + OPP_Perc_STL +'
                                    'L1_N_Awards_Won + L1_Coach_RS_W_Perc_Overall + L1_Coach_P_W_Perc +'
                                    'AVG_PLAYER_AGE + Coach_N_Seasons_Overall + Coach_Perc_Seasons_TEAM +'
                                    ' EntityEffects + TimeEffects', data = nba, drop_absorbed = True) 
twofe_results = twofe_reg.fit(cov_type = "clustered")
print(twofe_results.summary)

# F-Test Marginal Effect
ftest1 = f_test(onefe_results, twofe_results)

ftest2 = f_test(onefe_results1, twofe_results)

# one way random effect entity model
onere_reg = plm.RandomEffects.from_formula(formula = 'W ~ Perc_3PA + Perc_2PA + Perc_AST + Perc_STL +'
                                    'PFminusPFD + OPP_Perc_3PA + OPP_Perc_AST + OPP_Perc_STL +'
                                    'L1_N_Awards_Won + L1_Coach_RS_W_Perc_Overall + L1_Coach_P_W_Perc +'
                                    'AVG_PLAYER_AGE + Coach_N_Seasons_Overall + Coach_Perc_Seasons_TEAM', data = nba)
onere_results = onere_reg.fit(cov_type = "clustered")
print(onere_results.summary)

# one way random effect time model
teams = nba.reset_index()["TEAM"].unique()
teams = {team: team_index for team_index, team in enumerate(teams)}
nba2 = nba.reset_index()
nba2["TEAM"] = nba2.TEAM.apply(lambda x: teams[x])
nba2 = nba2.set_index(["Season", "TEAM"])
onere_reg1 = plm.RandomEffects.from_formula(formula = 'W ~ Perc_3PA + Perc_2PA + Perc_AST + Perc_STL +'
                                    'PFminusPFD + OPP_Perc_3PA + OPP_Perc_AST + OPP_Perc_STL +'
                                    'L1_N_Awards_Won + L1_Coach_RS_W_Perc_Overall + L1_Coach_P_W_Perc +'
                                    'AVG_PLAYER_AGE + Coach_N_Seasons_Overall + Coach_Perc_Seasons_TEAM', data = nba2)
onere_results1 = onere_reg1.fit(cov_type = "clustered")
print(onere_results1.summary)


# LM Test
def lik_test(resMod, unresMod):
    like_ratio = 2*(unresMod.loglik - resMod.loglik)
    print(f"Log-likelihood Ratio: {round(like_ratio, 2)},\n p-value: {round(1-chi2.cdf(like_ratio, 1), 2)},\n Distribution: Chi^2({1})")
    
lik_test(regMod, onere_results)

lik_test(regMod, onere_results1)

# Hausman test of FE vs. RE (Entity Effects)
b_fe_cov = onefe_results.cov
b_re_cov = onere_results.cov
# (I) find overlapping coefficients:
common_coef = list(set(onefe_results.params.index).intersection(onere_results.params.index))


# (II) calculate differences between FE and RE:
b_diff = np.array(onefe_results.params[common_coef] - onere_results.params[common_coef])
df = len(b_diff)
b_diff.reshape((df, 1))
b_cov_diff = np.array(b_fe_cov.loc[common_coef, common_coef] -
                      b_re_cov.loc[common_coef, common_coef])
b_cov_diff.reshape((df, df))

# (III) calculate test statistic:
stat = abs(np.transpose(b_diff) @ np.linalg.inv(b_cov_diff) @ b_diff)
pval = 1 - stats.chi2.cdf(stat, df)

print(f'stat: {stat}\n')
print(f'pval: {pval}\n')

# Hausman test of FE vs. RE (Time Effects)
b_fe_cov1 = onefe_results1.cov
b_re_cov1 = onere_results1.cov
# (I) find overlapping coefficients:
common_coef1 = list(set(onefe_results1.params.index).intersection(onere_results1.params.index))


# (II) calculate differences between FE and RE:
b_diff1 = np.array(onefe_results1.params[common_coef] - onere_results1.params[common_coef])
df1 = len(b_diff1)
b_diff1.reshape((df1, 1))
b_cov_diff1 = np.array(b_fe_cov1.loc[common_coef1, common_coef1] -
                      b_re_cov1.loc[common_coef1, common_coef1])
b_cov_diff1.reshape((df1, df1))

# (III) calculate test statistic:
stat = abs(np.transpose(b_diff1) @ np.linalg.inv(b_cov_diff1) @ b_diff)
pval = 1 - stats.chi2.cdf(stat, df)

print(f'stat: {stat}\n')
print(f'pval: {pval}\n')



# two way random effect model 
# import time

# start = time.time()


# def check_balanced(data, id_col, time_col):
#     data = data.sort_values([id_col, time_col])
#     N = data[id_col].nunique() 
#     T = data[time_col].nunique() 
    
#     obs_per_id = data.groupby(id_col).size() 
#     is_balanced = all(obs_per_id == T) 
#     if is_balanced: 
#         return data, N, T

# def feasible_gls(formula, data, max_iter=20, tol=1e-5, **kwargs): 
#     try: 
#         data, N, T = check_balanced(data=data, **kwargs) 
#     except TypeError:
#         print("Unbalanced Panel Data !") 
#         return
#     ols_model = smf.ols(formula=formula, data=data) 
#     ols_res = ols_model.fit(cov_type = "clustered") 
#     resid = ols_res.resid 
#     bar_J_T = 1/T * np.ones(shape=(T, T)) 
#     bar_J_N = 1/N * np.ones(shape=(N, N)) 
#     E_N = np.eye(N) - bar_J_N 
#     E_T = np.eye(T) - bar_J_T 
#     Q_1 = np.kron(E_N, E_T) 
#     Q_2 = np.kron(E_N, bar_J_T) 
#     Q_3 = np.kron(bar_J_N, E_T) 
#     Q_4 = np.kron(bar_J_N, bar_J_T)
    
#     last_beta_esti = np.zeros_like(ols_res.params) 
#     for i in range(1, max_iter + 1): 
#         w_1 = (resid @ Q_1 @ resid) / np.trace(Q_1) 
#         w_2 = (resid @ Q_2 @ resid) / np.trace(Q_2) 
#         w_3 = (resid @ Q_3 @ resid) / np.trace(Q_3) 
#         w_4 = w_2 + w_3 - w_1 
#         omega = w_1 * Q_1 + w_2 * Q_2 + w_3 * Q_3 + w_4 * Q_4 
        
#         gls_model = smf.gls(formula=formula, data=data, sigma=omega) 
#         gls_res = gls_model.fit(cov_type = "clustered") 
#         resid = gls_res.resid 
#         with np.printoptions(formatter={'float': '{:0.4f}'.format}): 
#             print(f">> Iteration {i}: beta = {gls_res.params.values}") 
    
#         if np.linalg.norm(last_beta_esti - gls_res.params) < tol or i == max_iter: 
#             print(">> Estimation Converged." if i < max_iter 
#                   else "Reached Maximum Iterations.") 
#             print(f">> w1: {w_1:.4f}, w2: {w_2:.4f}, w3: {w_3:.4f}, w4: {w_4:.4f}\n") 
#             break 
#         last_beta_esti = gls_res.params 

#     summ = pd.DataFrame(data={ 
#         'Parameter': round(last_beta_esti, 4),
#         'Std. Err.': round(gls_res.bse, 4), 
#         'T-stat': round(gls_res.tvalues, 4),
#         'P-value': round(gls_res.pvalues, 4) 
#         }) 
#     print(summ)
# feasible_gls('W ~ Perc_3PA + Perc_AST + Perc_STL +'
            #  'PFminusPFD + OPP_Perc_3PA + OPP_Perc_AST + OPP_Perc_STL +'
            #  'L1_N_Awards_Won + L1_Coach_RS_W_Perc_Overall + L1_Coach_P_W_Perc +'
            #  'AVG_PLAYER_AGE + Coach_N_Seasons_Overall + Coach_Perc_Seasons_TEAM +'
            #  'C(time) + C(indi)', nba, id_col = 'indi', time_col = 'time') 

   
