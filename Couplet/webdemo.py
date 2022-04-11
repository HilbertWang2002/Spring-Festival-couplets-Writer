import os
import sys
import torch
from flask import Flask, request, render_template
import argparse
from main import init_model_by_key
from module import Tokenizer, init_model_by_key

MODEL_PATH = sys.argv[1]

class Context(object):
    def __init__(self, path):
        path = MODEL_PATH
        print(f"loading pretrained model from {path}")
        self.device = torch.device('cpu')
        model_info = torch.load(path)
        self.tokenizer = model_info['tokenzier']
        self.model = init_model_by_key(model_info['args'], self.tokenizer)
        self.model.load_state_dict(model_info['model'])
        self.model.to(self.device)
        #self.model.eval()

    def predict(self, s):
        input_ids = torch.tensor(self.tokenizer.encode(s)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(input_ids).squeeze(0)
        pred = logits.argmax(dim=-1).tolist()
        pred = self.tokenizer.decode(pred)
        return pred
        
app = Flask(__name__)
ctx = Context(MODEL_PATH)

def generate_my_hash(c):
    result = ''
    count = 0
    my_dict = {}
    for i in c:
        if i not in my_dict.keys():
            result += str(count)
            my_dict[i] = count
            count += 1
        else:
            result += str(my_dict[i])
    return result
            

@app.route('/<coupletup>')
def api(coupletup):
    input_hash = generate_my_hash(coupletup)
    for i in range(3000):
        output = ctx.predict(coupletup)
        output_hash = generate_my_hash(output)
        if input_hash == output_hash and len(set(coupletup)&set(output))==0:
            #print(coupletup, input_hash, output, output_hash, len(set(coupletup)&set(output)))
            return output
    return 'Failed'
    


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template("index.html")
    coupletup = request.form.get("coupletup")
    coupletdown = ctx.predict(coupletup)
    return render_template("index.html", coupletdown=coupletdown)


if __name__ == '__main__':
    app.run(host='localhost', port=12345)
