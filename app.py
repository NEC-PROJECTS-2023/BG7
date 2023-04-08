import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, render_template
import pickle

app = Flask(__name__,template_folder="templates")
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/detect',methods=['GET','POST'])
def detect():
    return render_template('Detection.html')
@app.route('/riskpred',	methods=['GET','POST'])
def riskpred():
    return render_template('Risk_Prediction.html')




@app.route('/predict',methods=['POST'])
def predict():
    input_features = [float(x) for x in (request.form.values())]
	
    features_value = [np.array(input_features)]

    
    features_name = ['radius_mean', 'texture_mean', 'smoothness_mean',
       'compactness_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'smoothness_se', 'compactness_se',
       'symmetry_se', 'fractal_dimension_se']
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)  
    if output == 0:
        return render_template('failure_page.html')
    else : 
       return render_template('SUCCESS_PAGE2.html')

	
        

    #return render_template('index.html', prediction_text='Patient has {}'.format(res_val))

if __name__ == "__main__":
    app.run(debug=True)
    
