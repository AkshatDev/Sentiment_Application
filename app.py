from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from prediction_service import prediction
from pyngrok import ngrok
from flask_ngrok import run_with_ngrok

webapp_root = "webapp"

static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")

app = Flask(__name__, static_folder=static_dir,template_folder=template_dir)
run_with_ngrok(app)   
NGROK_AUTH_TOKEN = "224wCncH7aN2TJpFYf3cU2xafRQ_3QBGAkQXWGRM2L3rMytxR"
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":
        try:
            if request.form:
                dict_req = dict(request.form)
                response = prediction.form_response(dict_req)
                return render_template("index.html", response=response)
            #elif request.json:
                #response = prediction.api_response(request.json)
      
                #return jsonify(response)

        except Exception as e:
            print(e)
            error = {"error": "Something went wrong!! Try again later!"}
            error = {"error": e}

            return render_template("404.html", error=error)
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run()