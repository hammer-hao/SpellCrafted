from flask import Flask, request, make_response

app = Flask(__name__)

def response(message):
    resp = make_response(message)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

def generate_card(prompt):
    name = prompt+' Van Darkholme Name'
    cost = prompt+' Van Darkholme Cost'
    description = prompt+' My name is Van'
    return name, cost, description

@app.route('/')
def generate():
    prompt = request.args.get('prompt')
    name, cost, description = generate_card(prompt)
    text={'name': name ,
          'cost' : cost, 
          'description': description,
          }
    return response(text)

if __name__ == '__main__':
    app.run()
