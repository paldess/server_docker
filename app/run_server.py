# USAGE
# Start the server:
#       python run_front_server.py
# Submit a request via Python:
#       python simple_request.py

# import the necessary packages
import dill
import pandas as pd
import os
dill._dill._reverse_typemap['ClassType'] = type
#import cloudpickle
import flask
import logging
from logging.handlers import RotatingFileHandler
from time import strftime


# initialize our Flask application and the model
app = flask.Flask(__name__)

handler = RotatingFileHandler(filename='app.log', maxBytes=100000, backupCount=10)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

def load_model(model_path):
        with open(model_path, 'rb') as f:
                model_load = dill.load(f)
        return model_load

# modelpath = "/app/app/models/logreg_pipeline.dill"
# model = load_model(modelpath)

@app.route("/", methods=["GET"])
def general():
        return flask.jsonify("dataset download to https://archive.ics.uci.edu/ml/machine-learning-databases/00544/  (Gender-Age-Height-Weight-family_history_with_overweight-FAVC-FCVC-NCP-CAEC-SMOKE-CH2O-SCC-FAF-TUE-CALC-MTRANS")
        # return flask.jsonify("""Welcome to fraudelent prediction process. Please use 'http://<address>/predict' to POST""")

@app.route("/predict", methods=["POST"])
def predict():
        model_load = load_model('/app/app/model/model.pkl')
        model = model_load['model']
        threshold = model_load['threshold']
        data = pd.DataFrame([None], columns=['target'])
        if flask.request.method == "POST":
                try:
                        data = pd.read_json(flask.request.get_json())
                        data['target'] = [1 if i > threshold else 0 for i in model.predict(data)]
                except:
                        data = pd.DataFrame(['error! неверные данные', ], columns=['target'])
        out = data['target'].to_dict()
        return flask.jsonify(out)
# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
        print(("* Loading the model and Flask starting server..."
                "please wait until server has fully started"))
        port = int(os.environ.get('PORT', 8180))
        app.run(host='0.0.0.0', debug=True, port=port)
