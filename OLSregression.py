    # -*- coding: utf-8 -*-
"""
 Created on Wed Aug  2 18:08:08 2023
    
@author: be
    """
    
    # -*- coding: utf-8 -*-
"""
    Created on Mon Jul 24 14:00:24 2023
    @author: 16617
    """
    import pandas as pd
    import numpy as np
    #create DataFrame
    df = pd.DataFrame({'hours': [1, 2, 4, 5, 5, 6, 6, 7, 8, 10, 11, 11, 12, 12, 14],
    'score': [64, 66, 76, 73, 74, 81, 83, 82, 80, 88, 84, 82, 91,
    93, 89]})
    #view DataFrame
    print(df)
    import statsmodels.api as sm
    #define predictor and response variables
    y = df['score']
    x = df['hours']
    #add constant to predictor variables
    x = sm.add_constant(x)
    #fit linear regression model
    model = sm.OLS(y, x).fit()
    #view model summary
    print(model.summary())
    import matplotlib.pyplot as plt
    #find line of best fit
    a, b = np.polyfit(df['hours'], df['score'], 1)
    #add points to plot
    plt.scatter(df['hours'], df['score'], color='purple')
    #add line of best fit to plot
    plt.plot(df['hours'], a*df['hours']+b)
    #add fitted regression equation to plot
    plt.text(1, 90, 'y = ' + '{:.3f}'.format(b) + ' + {:.3f}'.format(a) + 'x', size=12)
    #add axis labels
    plt.xlabel('Hours Studied')
    plt.ylabel('Exam Score')
