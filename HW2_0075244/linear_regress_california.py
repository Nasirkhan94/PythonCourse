import pandas as pd
import numpy as np
import scipy.stats as sta
import matplotlib.pyplot as plt


def linear_regress(X=None, Y=None, data_set=None):

    data = pd.read_csv(data_set, sep="," , skiprows=0)
    print(data.head())

    # Drop the rows where at least one element is missing 
    A = data
    A.dropna()
    
    # Covert the data into a numpy array for convenience
    array_data = np.array(A.values, 'float')

    # Assign input and target variables
    Xa = array_data[:, 0]
    Ya = array_data[:,1]
    
    #  0.95  confidence interval for the sample data
    c_interval = sta.t.interval(alpha=0.95, df=len(Xa) - 1, loc=np.mean(Xa), scale=sta.sem(Xa))
    

    # Data standardization   
    X = Xa /(np.max(Xa))
    Y= Ya /1000.0 # Scaling the labels
    
   
    # Plot the data
    plt.plot(Xa, Y, 'co',)
    plt.ylabel("Median_house_value")
    plt.xlabel("Total_rooms")
    plt.title("Median_house_value VS Total_rooms")
    plt.grid()
    plt.show()
    
      # initialising parameter
    m = np.size(Y)
    X = X.reshape([len(X), 1])

    
    X = X.reshape([len(X), 1])
    x = np.hstack([np.ones_like(X), X])
    
    y_test = Y.reshape([len(X), 1])


    def computecost(x, y, beta):
 
        b = np.sum(((x @ beta) - Y) ** 2)
        cost_cal =  ( 1 / (2 * m)) * b
        return cost_cal

  
    beta = np.zeros([2, 1])

    # print(beta, '\n', m)

    # print(computecost(x, y, beta))

    def gradient_des(x, Y, beta,alpha,epochs):
       
        costg= np.zeros([epochs, 1])
        for iter in range(0, epochs):
            error = (x @ beta) - Y
            w = beta[0] - ((alpha / m) * np.sum(error * x[:, 0]))
            b = beta[1] - ((alpha / m) * np.sum(error * x[:, 1]))
            beta = np.array([w, b]).reshape(2, 1)
            costg[iter] = (1 / (2 * m)) * (np.sum((error) ** 2))  
        return beta, costg
   
    alpha = 0.0001
    epochs = 1000
    
    beta,costg= gradient_des(x, Y, beta,alpha,epochs)
    
  ## Cost function with epochss
    k=np.arange(epochs);
    plt.plot(k,costg)
    plt.xlabel("Number of epochs")
    plt.ylabel("Cost J(theta))")
    plt.show()


    regression_estimates = x @ beta
   
    residual_errors = regression_estimates - y_test
  
 
    #print("The residuals for the regression : ")
          
    #print(residual_errors)
    
    #print("The standard error for the regression:")
    
    str_error_regression=np.sqrt( (1 / (m-2)) * np.sum(( residual_errors) ** 2)  )
 
    

    # plot linear fit for our beta
   
    plt.plot(Xa, Y, 'bo')
    plt.plot(Xa, x @ beta, 'r-')
    plt.ylabel("Median_house_value")
    plt.xlabel("Total_rooms")
    plt.title("Median_house_value VS Total_rooms")
  
    plt.legend(['Points', 'Linear-Regression Fit'])
    plt.grid()
    plt.show()

    return regression_estimates,beta, residual_errors,str_error_regression ,c_interval
