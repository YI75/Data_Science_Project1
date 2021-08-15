#Can we predict a song's popularity based on its danceabilty, energy, and duration?
from math import remainder
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

#Read csv file into variable genres_df
genres_df = pd.read_csv('data_by_genres_o.csv')

#Rows containing missing values are filtered out 
genres_df = genres_df.dropna(axis=0)

#Tranform the time in the duration column from milliseconds to minutes
genres_df = genres_df.rename(columns={"duration_ms": "duration_min"})
genres_df.duration_min = genres_df.duration_min.map(lambda time: time*0.001*(1/60))

#Select data for modeling
y_populartiy = genres_df.popularity

X_features = genres_df[['danceability','energy','duration_min']]

#Split data to avoid underfitting and overfitting
train_X_features, val_X_features, train_y_popularity, val_y_popularity = train_test_split(X_features,y_populartiy)

#Building the Model, a Random Forest Regressor
forest_model = RandomForestRegressor(random_state=1)

forest_model.fit(train_X_features,train_y_popularity)

song_pop_preds = forest_model.predict(val_X_features)

#Print out mean absolute error to test the model's accuracy
print("The Mean Absolute Value is: ", mean_absolute_error(val_y_popularity,song_pop_preds))

#Print out the predicted popularity of songs
print(pd.DataFrame({'Predicted Popularity':song_pop_preds}))








 
