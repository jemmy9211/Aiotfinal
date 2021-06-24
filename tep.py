# base
import pandas as pd
  
# stockdata

from scipy.stats import linregress
 
# visual
import matplotlib.pyplot as plt

Y=[574.6842651367188,577.9666137695312,572.4092407226562,567.5690307617188,574.3389892578125,570.8012084960938,573.7166748046875,572.759033203125,572.5528564453125,576.0073852539062]
X=[1,2,3,4,5,6,7,8,9,10]
reg_up = linregress(x = X,y = Y)
print(reg_up.slope)