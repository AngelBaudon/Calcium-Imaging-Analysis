# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 10:38:30 2021

@author: Angel.BAUDON


To do on this code :
    - WARNING ! Every pval are corrected with yhe bonferoni method wich is the most conservative !
    - QQ plots (& unbiased methods to estimate the fit ?)
    - Test sphericity ?
    
"""
def IntraGrpStat(df, Paired=False):
    '''
    Parameters
    ----------
    df: pd.DataFrame that contain at least:
            - a column named 'Values' containing dependant variables
            - a column names 'Group' containing group names
            
        If the data are paired:
            - a column named 'Subject' containing Subject ID
            - a column 'Time' containing the different time points
    
    Paired: BOOL, optional. The default is False.
    
    Returns
    -------
    Output: pd.DataFrame.
    
    '''
    import scipy, numpy as np, pandas as pd
    from statsmodels.stats.anova import AnovaRM
    from scikit_posthocs import posthoc_dunn, posthoc_conover, posthoc_tukey
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    

    
    '''               Test the criteria for parametric tests         '''
    norm, residuals, grps = [], [], []
    for g in list(set(df['Group'])):
        
        if not 'Time' in df:
            data = df[df['Group'] == g]['Values']
            residuals.extend([x-np.nanmean(data) for x in data])
            grps.append(data)
             
        else:
            dfgrp = df[df['Group'] == g]
            for t in list(set(dfgrp['Time'])):
                datime = dfgrp[dfgrp['Time'] == t]['Values']
                residuals.extend([x-np.nanmean(datime) for x in datime])
                grps.append(datime)
    
    #Normality
    S, p_val_s = scipy.stats.shapiro(residuals)
    norm.append(False) if p_val_s < 0.05 else norm.append(True)
    
    #Equality of variances
    L, p_val_l = scipy.stats.levene(*grps)
    norm.append(False) if p_val_l < 0.05 else norm.append(True)
    
    print(norm)
    
    
    '''                                   Decision tree                 '''
    groups = [df[df['Group'] == x]['Values'] for x in set(df['Group'])]
    
    #T-test familly
    if len(groups) == 2:
        #Parametric test
        if not False in norm:
            if Paired:
                stat, pval = scipy.stats.ttest_rel(*groups)
                test = 'Paired t-Test'
            else:
                stat, pval = scipy.stats.ttest_ind(*groups)
                test = 'Unpaired t-Test'
                
                
        #Non parametric test
        if False in norm:
            if Paired:
                stat, pval = scipy.stats.wilcoxon(*groups)
                test = 'Wilcoxon'
            else:
                stat, pval = scipy.stats.mannwhitneyu(*groups)
                test = 'Mann-Whitney'
          
    
    #Anova familly
    elif len(groups) > 2:        
        if Paired:
            if not False in norm:
                aovrm = AnovaRM(df, depvar='Values', subject='Subject', within=['Time'])
                res = aovrm.fit().summary().tables[0]
                stat, pval = float(res['F Value']), float(res['Pr > F'])
    
                ph = pairwise_tukeyhsd(df['Values'], df['Time'])
                ph = pd.DataFrame(data=ph._results_table.data[1:],
                                  columns=ph._results_table.data[0])
                ph_out = {'Test': 'RM ANOVA & paired Tukey'}
                for x, y, z in zip(*[list(ph[x]) for x in ['group1', 'group2', 'p-adj']]):
                    ph_out[f'{x} vs {y}'] = z
    
            
            else: 
                stat, pval = scipy.stats.friedmanchisquare(*groups)
                ph = posthoc_conover(groups, p_adjust = 'bonferroni')
                print(ph)
                
                ph_out = {'Test': 'Friedman & Wilcoxon with Bonferoni correction'}
                # for x, y, c1, c2 in zip([1,1,2], [2,3,3], comp1, comp2):
                #     ph_out[f'{c1} vs {c2}'] = float(ph.loc[x, y])
        
        else:
            if not False in norm:
                stat, pval = scipy.stats.f_oneway(*groups)
                ph = posthoc_tukey(df, val_col='Values', group_col='Time')
                
                ph_out = {'Test': 'One-Way ANOVA & Tukey'}
                # for x, y in zip(comp1, comp2):
                #     ph_out[f'{x} vs {y}'] = float(ph.loc[x, y])
                    
            else:
                stat, pval = scipy.stats.kruskal(*groups)
                ph = posthoc_dunn(groups, p_adjust = 'bonferroni')
                
                ph_out = {'Test': "Kruskal-Wallis & Dunn's MC"}
                # for x, y, c1, c2 in zip([1,1,2], [2,3,3], comp1, comp2):
                #     ph_out[f'{c1} vs {c2}'] = float(ph.loc[x, y])
    
    try: return(round(stat, 4), round(pval, 4), ph_out, ph)
    except NameError: return(round(stat, 4), round(pval, 4), test)
