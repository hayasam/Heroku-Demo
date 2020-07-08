import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
modelLR = pickle.load(open('LinearRegression.pkl', 'rb'))
modelDT = pickle.load(open('DecisionTreeRegression.pkl', 'rb'))
modelRF = pickle.load(open('RandomForestRegression.pkl', 'rb'))
modelNN = pickle.load(open('NeuralNetworks_50.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    predictionLR = modelLR.predict(final_features)
    predictionDT = modelDT.predict(final_features)
    predictionRF = modelRF.predict(final_features)
    NN_int_features = [predictionLR[0][0], predictionDT[0],predictionRF[0]]
    NN_final_features = [np.array(NN_int_features)]
    predictionNN= modelNN.predict(NN_final_features)


    # output = round(predictionNN[0], 2)
    output = np.round(predictionNN[0], 2)

    return render_template('index.html', prediction_text='PCI is = $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
