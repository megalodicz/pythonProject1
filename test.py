import pandas as pd
data = pd.read_csv('cow_beef.csv')
data = pd.DataFrame(data)
X = data.drop(columns='price', axis=1)
Y = data['price']
print(Y)