import pandas as pd
# Create the pandas dataframe for the data

weather_data_dictionary = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Windy': [False, True, False, False, False, True, True, False, False, False, True, True, False, True],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

# Create the DataFrame
weather_dataframe = pd.DataFrame(weather_data_dictionary)
# print(weather_dataframe)

#Perform Mapping to Numerical values
weather_dataframe['Outlook'] = weather_dataframe['Outlook'].map({'Sunny':0,'Overcast':1,"Rain":2})
weather_dataframe['Temperature'] = weather_dataframe['Temperature'].map({'Hot': 0, 'Mild': 1, 'Cool': 2}) 
weather_dataframe['Humidity'] = weather_dataframe['Humidity'].map({'High': 0, 'Normal': 1}) 
weather_dataframe['Windy'] = weather_dataframe['Windy'].map({False: 0, True: 1}) 
weather_dataframe['PlayTennis'] = weather_dataframe['PlayTennis'].map({'No': 0, 'Yes': 1})

print(weather_dataframe['PlayTennis'].value_counts)