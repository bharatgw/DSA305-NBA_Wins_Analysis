import pandas as pd 
import numpy as np
import scipy.stats as stats
import linearmodels as plm 
from linearmodels.panel import PooledOLS
import matplotlib.pyplot as plt


nba = pd.read_csv("data/finaldf.csv")
nba = nba.dropna()

time = pd.Categorical(nba.Season)
indi = pd.Categorical(nba.TEAM)
nba = nba.set_index(["TEAM", "Season"])
nba["time"] = time
nba["indi"] = indi 
nba = nba.drop(columns = ["Perc_2PA", 'Coach'])




# One way entity fixed effect model 
onefe_reg = plm.PanelOLS.from_formula(formula = 'W ~ Perc_3PA + Perc_AST + Perc_STL +'
                                    'PFminusPFD + OPP_Perc_3PA + OPP_Perc_AST + OPP_Perc_STL +'
                                    'L1_N_Awards_Won + L1_Coach_RS_W_Perc_Overall + L1_Coach_P_W_Perc +'
                                    'AVG_PLAYER_AGE + Coach_N_Seasons_Overall + Coach_Perc_Seasons_TEAM +'
                                    'C(indi) + EntityEffects', data = nba, drop_absorbed = True) 
onefe_results = onefe_reg.fit()
print(onefe_results.summary)
# One way time fixed effect mode 
onefe2_reg = plm.PanelOLS.from_formula(formula = 'W ~ Perc_3PA + Perc_AST + Perc_STL +'
                                    'PFminusPFD + OPP_Perc_3PA + OPP_Perc_AST + OPP_Perc_STL +'
                                    'L1_N_Awards_Won + L1_Coach_RS_W_Perc_Overall + L1_Coach_P_W_Perc +'
                                    'AVG_PLAYER_AGE + Coach_N_Seasons_Overall + Coach_Perc_Seasons_TEAM +'
                                    'C(time) + TimeEffects', data = nba, drop_absorbed = True) 
onefe2_results = onefe2_reg.fit()
print(onefe2_results.summary)
# Two way fixed effect model
twofe_reg = plm.PanelOLS.from_formula(formula = 'W ~ Perc_3PA + Perc_AST + Perc_STL +'
                                    'PFminusPFD + OPP_Perc_3PA + OPP_Perc_AST + OPP_Perc_STL +'
                                    'L1_N_Awards_Won + L1_Coach_RS_W_Perc_Overall + L1_Coach_P_W_Perc +'
                                    'AVG_PLAYER_AGE + Coach_N_Seasons_Overall + Coach_Perc_Seasons_TEAM +'
                                    'C(indi) + C(time) + EntityEffects + TimeEffects', data = nba, drop_absorbed = True) 
twofe_results = twofe_reg.fit()
print(twofe_results.summary)

# one way random effect model 
onere_reg = plm.RandomEffects.from_formula(formula = 'W ~ Perc_3PA + Perc_AST + Perc_STL +'
                                    'PFminusPFD + OPP_Perc_3PA + OPP_Perc_AST + OPP_Perc_STL +'
                                    'L1_N_Awards_Won + L1_Coach_RS_W_Perc_Overall + L1_Coach_P_W_Perc +'
                                    'AVG_PLAYER_AGE + Coach_N_Seasons_Overall + Coach_Perc_Seasons_TEAM +'
                                    'C(time)', data = nba)
onere_results = onere_reg.fit()
print(onere_results.summary)


# two way random effect model 
def check_balanced(data, id_col, time_col):
    data = data.sort_values([id_col, time_col])
    N = data[id_col].nunique() 
    T = data[time_col].nunique() 
    return data, N, T
    
def feasible_gls(formula, data, max_iter=20, tol=1e-5, **kwargs): 
    data, N, T = check_balanced(data=data, **kwargs) 
    return 
    N = data[id_col].nunique() 
    T = data[time_col].nunique() 
    ols_model = smf.ols(formula=formula, data=data) 
    ols_res = ols_model.fit() 
    resid = ols_res.resid 
    bar_J_T = 1/T * np.ones(shape=(T, T)) 
    bar_J_N = 1/N * np.ones(shape=(N, N)) 
    E_N = np.eye(N) - bar_J_N 
    E_T = np.eye(T) - bar_J_T 
    Q_1 = np.kron(E_N, E_T) 
    Q_2 = np.kron(E_N, bar_J_T) 
    Q_3 = np.kron(bar_J_N, E_T) 
    Q_4 = np.kron(bar_J_N, bar_J_T)
    
    last_beta_esti = np.zeros_like(ols_res.params) 
    for i in range(1, max_iter + 1): 
        w_1 = (resid @ Q_1 @ resid) / np.trace(Q_1) 
        w_2 = (resid @ Q_2 @ resid) / np.trace(Q_2) 
        w_3 = (resid @ Q_3 @ resid) / np.trace(Q_3) 
        w_4 = w_2 + w_3 - w_1 
        omega = w_1 * Q_1 + w_2 * Q_2 + w_3 * Q_3 + w_4 * Q_4 
        
        gls_model = smf.gls(formula=formula, data=data, sigma=omega) 
        gls_res = gls_model.fit() 
        resid = gls_res.resid 
        with np.printoptions(formatter={'float': '{:0.4f}'.format}): 
            print(f">> Iteration {i}: beta = {gls_res.params.values}") 
    
        if np.linalg.norm(last_beta_esti - gls_res.params) < tol or i == max_iter: 
            print(">> Estimation Converged." if i < max_iter 
                  else "Reached Maximum Iterations.") 
            print(f">> w1: {w_1:.4f}, w2: {w_2:.4f}, w3: {w_3:.4f}, w4: {w_4:.4f}\n") 
            break 
        last_beta_esti = gls_res.params 

    summ = pd.DataFrame(data={ 
        'Parameter': round(last_beta_esti, 4),
        'Std. Err.': round(gls_res.bse, 4), 
        'T-stat': round(gls_res.tvalues, 4),
        'P-value': round(gls_res.pvalues, 4) 
        }) 
    print(summ)


feasible_gls(formula = 'W ~ Perc_3PA + Perc_AST + Perc_STL +'
                                    'PFminusPFD + OPP_Perc_3PA + OPP_Perc_AST + OPP_Perc_STL +'
                                    'L1_N_Awards_Won + L1_Coach_RS_W_Perc_Overall + L1_Coach_P_W_Perc +'
                                    'AVG_PLAYER_AGE + Coach_N_Seasons_Overall + Coach_Perc_Seasons_TEAM',
                                    data = nba, id_col = 'indi', time_col = 'time')

