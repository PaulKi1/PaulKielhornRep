import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

df = pd.read_excel(r"C:")

# Strike wurde gegen Moneyness ausgetauscht, damit ist benutzbar für den großen Code aufm PC ist. Die oben verlinkte
# Excel Datei funktioniert folglich nicht mehr

moneyness = df['Moneyness']                                  # This creates a new dataframe for the moneyness
unique_moneyness = np.unique(moneyness)                      # This creates a numpy array for the unique moneyness
moneyness_columns = pd.DataFrame(columns=unique_moneyness)    # This creates a new df with the columns sorted from low to high and only the unique ones
print(moneyness_columns)

#moneyness_columns = pd.DataFrame(sorted(df['moneyness'].unique(), reverse=False))     # This does the same thing, but creates a list, unless you put pd.DataFrame in front
#print(moneyness_columns)

#if new_df.columns.equals(moneyness_df.columns):          # This checks if two dataframes have the same columns
#    print("Same names")
#else:
#    print("Different names")

quote_datetime = df['quote_datetime']
unique_time = pd.DataFrame({'Quote_Time': df['quote_datetime'].drop_duplicates()})
print(unique_time)

cor_df = pd.DataFrame(columns=moneyness_columns)

values_df = df[['Moneyness', 'gamma', 'quote_datetime']]

for col in moneyness_columns:
    for index, row in unique_time.iterrows():
        time = row['Quote_Time']
        # Filter values_df to get gamma values at the specified time and moneyness column
        gamma_value = values_df.loc[(values_df['quote_datetime'] == time) & (values_df['Moneyness'] == float(col)), 'gamma'].values[0]
        cor_df.at[index, col] = gamma_value

print(cor_df)
correlation_matrix = cor_df.corr()
print(correlation_matrix)
dataplot = sb.heatmap(cor_df.corr(), cmap="YlGnBu", annot=True)
plt.show()


for col in moneyness_columns.columns:
    col_data = []  # Create an empty list to store the column data
    for index, row in unique_time.iterrows():
        time = row['Quote_Time']
        # Filter values_df to get gamma values at the specified time and moneyness column
        gamma_value = values_df.loc[(values_df['quote_datetime'] == time) & (values_df['moneyness'] == float(col)), 'gamma'].values[0]
        col_data.append(gamma_value)  # Append the gamma_value to the list

    # Create a new DataFrame for the current column data
    col_df = pd.DataFrame({col: col_data})
    cor_df = pd.concat([cor_df, col_df], axis=1)  # Concatenate along the columns axis


