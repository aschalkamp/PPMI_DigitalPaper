import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def clean_hc(hc,covariates=['age','male'],target=''):
    hc = hc.dropna(subset=np.hstack([covariates,target]),how='any',axis='rows')
    return hc

def fit_model(data,formula=' ~ male + age'):
    mod = smf.ols(formula=formula, data=data)
    res = mod.fit()
    print(res.summary())
    return res

def get_residuals(model,data,res_name="average acceleration residual",covariates=['age','male','BMI'],target='',intercept=True):
    params = model.params
    data['pred'] = 0
    for cov in covariates:
        data['pred'] += params[cov] * data[cov]
    if intercept:
        data['pred'] += params['Intercept']
    data[res_name] = data[target] - data['pred']
    return data

def fit_get_res(hc,data,target='',covariates=['age','male'],res_name="average acceleration residual_bmi",
                save='/scratch/c.c21013066/data/ppmi/analyses/studywatch/hc_residuals',intercept=True):
    formula = f'{target} ~ ' + ' + '.join(covariates)
    res = fit_model(hc,formula)
    if len(save)>1:
        res.save(f"{save}/{res_name}_model.pickle")
    data = get_residuals(res,data,res_name,covariates,target,intercept=intercept)
    return res,data