from flask import Flask, render_template, redirect, url_for, request, redirect, g
from generator.generator import generate

import torch
import torch.nn as nn
from tokenizers import Tokenizer
from torch.nn import functional as F

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method=="POST":
        if len(request.form['prompt'])==0:
            return render_template('index.html', response='[empty]')
        else:
            response=generate(request.form['prompt'])
            return render_template('index.html', response=response)
    else:
        return render_template('index.html', response='[empty]')
    
if __name__=='__main__':
    app.run()