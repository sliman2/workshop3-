from flask import Flask
from models.model1 import model1
#from models.model2 import model2
#etc...
from routes import api
app = Flask(__name__)
app.register_blueprint(api)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)