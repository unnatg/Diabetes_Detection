import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tkinter import *


data=pd.read_csv('diabetes.csv')
data.head()
y=data['Outcome']
data.drop(['Outcome'],axis=1,inplace=True)
data.head()

x_train, x_test, y_train, y_test = train_test_split(data.values, y, test_size = 0.3, random_state = 42)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)

root = Tk()

root.title("Diabetes Predictor")
#root.iconbitmap('Z:\Study\Second Year\AI\GUI\icon.ico')

label1 = Label(root, text="Please fill out the queries below to know whether you are diabetic or not")
#pregnancy parameter
label1.grid(row=0,column=0)
label2 = Label(root, text="Are you Pregnant?")
label2.grid(row=1,column=0)
enter_preg = Entry(root, width=35, borderwidth=5)
enter_preg.grid(row=2, column=0, columnspan=3, padx=10, pady=10)

#Glucose parameter
label3= Label(root, text="Enter your glucose level")
label3.grid(row=3,column=0)
enter_glucose = Entry(root, width=35, borderwidth=5)
enter_glucose.grid(row=4, column=0, columnspan=3, padx=10, pady=10)

#BloodPressure Parameter
label4= Label(root, text="Enter your Blood Pressure")
label4.grid(row=5,column=0)
enter_bp = Entry(root, width=35, borderwidth=5)
enter_bp.grid(row=6, column=0, columnspan=3, padx=10, pady=10)

#Skin Thickness Parameter
label5= Label(root, text="Enter your Skin Thickness")
label5.grid(row=7,column=0)
enter_skin = Entry(root, width=35, borderwidth=5)
enter_skin.grid(row=8, column=0, columnspan=3, padx=10, pady=10)

#Insulin Parameter
label6= Label(root, text="Enter your Insulin level")
label6.grid(row=9,column=0)
enter_insulin = Entry(root, width=35, borderwidth=5)
enter_insulin.grid(row=10, column=0, columnspan=3, padx=10, pady=10)

##BMI Parameter
label6= Label(root, text="Enter your BMI ")
label6.grid(row=11,column=0)
enter_bmi = Entry(root, width=35, borderwidth=5)
enter_bmi.grid(row=12, column=0, columnspan=3, padx=10, pady=10)

##Diabetes Pedigree Function Parameter
label7= Label(root, text="Enter your Diabetes Predigree Function level")
label7.grid(row=13,column=0)
enter_func = Entry(root, width=35, borderwidth=5)
enter_func.grid(row=14, column=0, columnspan=3, padx=10, pady=10)

#Age Parameter
label8= Label(root, text="Enter your Age")
label8.grid(row=15,column=0)
enter_age = Entry(root, width=35, borderwidth=5)
enter_age.grid(row=16, column=0, columnspan=3, padx=10, pady=10)

def prediction():
    preg=int(enter_preg.get())
    glucose=int(enter_glucose.get())
    bp=int(enter_bp.get())
    skin_thickness=int(enter_skin.get())
    insulin=int(enter_insulin.get())
    bmi=int(enter_bmi.get())
    diabetes_function=float(enter_func.get())
    age=int(enter_age.get())
    predict_array=np.array([preg,glucose,bp,skin_thickness,insulin,bmi,diabetes_function,age]).astype(np.float64)
    predict=model.predict([predict_array])
    #print(predict)
    global result
    if(predict==1):
        final_label = Label(root, text="You have Diabetes").grid(row=18, column=0)
    else:
        final_label = Label(root, text="You don't have Diabetes").grid(row=18, column=0)

    accuracy()
    #label_acc = Label(root, text=acc).grid(row=19, column=0)

def accuracy():
    prediction = model.predict(x_test)
    global accuracy
    accuracy = accuracy_score(y_test, prediction)
    global acc
    #acc=("The accuracy of the classifier is =",accuracy*100,"%")

evaluate = Button(root, text="Evaluate", padx=40,pady=20, command=prediction)
evaluate.grid(row=17,column=0)
"""
label_final = Label(root, text=evaluate)
label_final.grid(row=18, column=0)
"""
"""Exit Button"""
button_quit = Button(root, text="Exit App", command=root.quit).grid(row=19,column=0)
root.mainloop()
