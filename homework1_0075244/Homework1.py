#!/usr/bin/env python
# coding: utf-8

# ##  Software for a Financial Institution to Model  their Clients’ Portfolio

# In[90]:



import random

class Portfolio:
    
    def __init__(self):
    
        self.cash = 0
        self.stock_dict = {}
        self.fund_dict ={}
        self.History=[]
        self.History_stock=[]
        self.total_MF = {}
        self.History_MF=[]
        
    def __str__(self):
        return  'Current Portfolio: \n Net cash in (USD) : {self.cash} \n Stocks history:, {self.History_stock} \n Mutual Funds history: {self.History_MF}'.format(self=self)
       
        
        
    def addcash(self, amount):
        self.cash = self.cash + amount
        p=print('Total amount after cash deposit of {fname} '.format(fname = amount))
        self.History.append(self.cash )
        return self.cash 
    
    def withdrawCash(self, amount):
        self.cash = self.cash - amount
        q= print('Total amount after cash withdrawal of {fname}'.format(fname = amount))
        self.History.append(self.cash )
        return self.cash 
    
    def new_stock(self, price , symbol):
        self.stock_dict[symbol] = price
       # return self.stock_dict
    
    
    def buy_stock(self, num, stk_symbol):
        
        for symbol, price in self.stock_dict.items():  
            if symbol == stk_symbol:
                #newval= list(self.stock_dict.values())
                 
                stockvalue = (num* price)
                self.cash = self.cash - stockvalue
                r= print('Total amount after purchasing {fname} stocks of {s} with price ${p} per stock ' .format(fname = num, s = stk_symbol ,p=price))
                self.History.append(self.cash )
                num=+ num
                self.History_stock.append(num)
                self.History_stock.append(symbol)
                return self.cash
      
        
          
    def sell_stock(self, num , stk_symbol):
        
         for symbol, price in self.stock_dict.items():  
            if symbol == stk_symbol:
                #newval= list(self.stock_dict.values())
                 
                selling_price=  (random.uniform(0.5*price, 1.5*price)) ## selling unform RV BW[0.5X,1.5X] 
                stockvalue = (num* selling_price)
                self.cash = self.cash + stockvalue
                s= print('Total amount after selling {fname} stocks of {s} ' .format(fname = num, s = stk_symbol))
                num=+ num
                num= -1*num
                self.History_stock.append(num)
                self.History_stock.append(symbol)
                self.History.append(self.cash )    
                return self.cash 
            
      
    def new_MFUND(self, price , symbol):
        self.fund_dict[symbol] = price
        #return self.fund_dict
    
    

    def buy_MFUND(self, num, stk_symbol):
        
        for symbol, price in self.fund_dict.items():  
            if symbol == stk_symbol:
                #newval= list(self.stock_dict.values())
                 
                stockvalue = (num* price)
                self.cash = self.cash - stockvalue
                t= print('Total amount after purchasing {fname} mutual funds of {s} of worth $ {p} per MF ' .format(fname = num, s = stk_symbol, p=price))
                self.History.append(self.cash )
                num=+ num
                self.History_MF.append(num)
                self.History_MF.append(symbol)
               
                return self.cash

    
    def sell_MFUND(self, num , stk_symbol):
        
         for symbol, price in self.fund_dict.items():  
            if symbol == stk_symbol:
                #newval= list(self.stock_dict.values())
                 
                selling_price=  (random.uniform(0.9, 1.2)) ## selling unform RV btw[0.9,1.2] 
                stockvalue = (num* selling_price)
                self.cash = self.cash + stockvalue
                u= print('Total amount after selling {fname} mutual fund of {s} ' .format(fname = num, s = stk_symbol))
                self.History.append(self.cash )
                num=+ num
                num= -1*num
                self.History_MF.append(num) 
                self.History_MF.append(symbol)
                return self.cash
   
    def history(self):
       # for x in range(len(self.History)):
        #    print( self.History[x])
        print("Cash amount(in (USD)) ordered by time",str(self.History) )
        print(' \n ')
        print('Transaction history:\n  ')
        print("Net cash available in in (USD):", self.History[-1] )
        print("Stocks history:", self.History_stock )
        print("Mutual Funds history:", self.History_MF )
       

        
   
        


# In[ ]:





# In[91]:


# Test1 portfolio
portfolio=Portfolio()
print(portfolio.addcash(300.50))
print(portfolio.new_stock(20,'HFH'))
print(portfolio.buy_stock(5,'HFH'))

print(portfolio.new_MFUND(1,'BRT'))
print(portfolio.new_MFUND(1,'GHT'))
print(portfolio.buy_MFUND(10.3,'BRT'))
print(portfolio.buy_MFUND(2,'GHT'))

print(portfolio.sell_stock(1,'HFH'))
print(portfolio.sell_MFUND(3,'BRT'))
print(portfolio.withdrawCash(50))
 
print(portfolio.history() )
 
#print(port1.portfolio() )

print(' \n ')
print(portfolio )



# In[93]:


port2=Portfolio()
print(port2.addcash(1000))

print(port2.new_stock(30,'Tesla'))


print(port2.buy_stock(3,'Tesla'))

print(port2.new_stock(50,'FB'))

print(port2.buy_stock(2,'FB'))

print(port2.addcash(500))

print(port2.new_stock(80,'GUCCI'))

print(port2.buy_stock(2,'GUCCI'))

print(port2.withdrawCash(200))

print(port2.sell_stock(2,'FB'))

print(port2.sell_stock(5,'FB'))


print(port2.new_MFUND(1,'GHT'))


print(port2.withdrawCash(300))



print(port2.new_MFUND(1,'BRT'))

print(port2.buy_MFUND(20.5,'GHT'))

print(port2.sell_MFUND(10.3,'BRT'))

print(port2.buy_MFUND(12.6,'GHT'))



print(port2.sell_MFUND(11.5,'GHT'))

print(' \n ') 
print(port2.history() )

print(' \n ')
print(portfolio )


# In[ ]:








