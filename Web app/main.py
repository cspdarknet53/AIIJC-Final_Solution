from flask import Flask, render_template, request
from inference import prep,load_json_file, SignsClassifier
import cv2
import numpy as np
import torch
from torch import nn
from PIL import Image

C2LP = 'app/class2label.json'
EN2DESC = 'app/desc_en.json'  
RU2DESC = 'app/desc_ru.json' 
MW = 'app/best.pth'  

class2label = load_json_file(C2LP)
class2name_en = load_json_file(EN2DESC)
class2name_ru = load_json_file(RU2DESC)

app = Flask(__name__) 

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file') # Extracting The File From Storage
        model = SignsClassifier('efficientnet', len(class2label)) # Model Input
        sdict = torch.load(MW, map_location='cpu')
        model.load_state_dict(sdict['state_dict'])  # Loading 'best.pth'
        model.eval() # Setting The Module Into The Evaluation Mode
        label2class = {v : k for k, v in class2label.items()} # Converting Prediction Into Actual Class Labels
        img = Image.open(file) # Opening The File Using PIL
        img = np.array(img) # Converting The Image To A NumPy Array
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB) # Converting Image From One Color Space To Another
        im = prep(img, (224, 224)) # Applying The Prep Function
        im = torch.from_numpy(im) # NumPy Array -> Tensor
        im = im.unsqueeze(0)
        pred = model(im) # Applying The Earlier Model Variable
        pred = nn.LogSoftmax(dim=1)(pred) # Using SoftMax Activation
        pred = pred.argmax(dim=-1).numpy()[0]
        pred = label2class[pred] # Converting Prediction To Class ID
        en_pred = class2name_en[pred] # Returning Prediction In English 
        ru_pred = class2name_ru[pred] # Returning Prediction In Russian 
        return render_template('result.html', pred_en=en_pred, classid=pred, pred_ru=ru_pred) # Applying The Prediction To The HTML Template
    return render_template('index.html') # HomePage
