from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open("model.pkl" , "rb") )

model_columns= ['Id', 'Sepal-Length', 'Sepal-Width', 'Petal-Length', 'Petal-Width',
       'Species']

app = Flask(__name__)




@app.route("/")
def home():
    return render_template("index.html", title="Homepage", features = model_columns)

@app.route("/output", methods=["POST"])
def output():
    input1 = request.form.get('input_1')
    input2 = request.form.get('input_2')
    input3 = request.form.get('input_3')
    input4 = request.form.get('input_4')

    arr = np.array([[input1,input2, input3, input4]])
    pred = model.predict(arr)
    return render_template("output.html", 
                             title= "Results", 
                             data= pred,
                             in1=input1, 
                             in2=input2, 
                             in3= input3, 
                             in4 =input4,
                             features = model_columns)




if __name__ == "__main__":
    app.run(debug=True, port = 9000)


