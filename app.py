import numpy as np
from flask import Flask, render_template,request
import pickle#Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('finalized_model.pkl', 'rb'))
scaler=pickle.load(open('scaler.pkl','rb'))
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    #For rendering results on HTML GUI
    int_features = [float(x) for x in request.form.values()]
    final_features = np.array(int_features)
    x=np.expand_dims(final_features,axis=0)
    x_test_scaled=scaler.transform(x)
    prediction = model.predict(x_test_scaled)
    output=prediction[0]
    f=float(output)
    g=round(f,2)
    return render_template('second.html', prediction_text='{} MPa'.format(g))
if __name__ == "__main__":
    app.run(debug=True)
