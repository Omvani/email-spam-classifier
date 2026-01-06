from preprocessing import load_and_preprocess_data
from sklearn.linear_model import LogisticRegression

x_train,x_test,y_train,y_test=load_and_preprocess_data()

model=LogisticRegression()

model.fit(x_train,y_train)

y_pred=model.predict(x_test)

print("model training completed.")

