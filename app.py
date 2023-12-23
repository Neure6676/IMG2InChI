import cv2
import torch
from model_tit import *
import util
import requests
from PIL import Image
from io import BytesIO
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from mangum import Mangum


app=FastAPI()
handler=Mangum(app)

image_size = 224

def read_image_from_url(url):
    try:
        # Send an HTTP request to the URL to get the image content
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Open the image using PIL
            img = Image.open(BytesIO(response.content))
            return img
        else:
            print(f"Failed to fetch image. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    


# start here ! ###################################################################################
def do_predict(net, tokenizer, image_file):

    text = []

    image = read_image_from_url(image_file)
    #image = cv2.imread(image_file,cv2.IMREAD_GRAYSCALE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, dsize=(image_size,image_size), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255
    image = torch.from_numpy(image).unsqueeze(0).repeat(1,3,1,1)

    net.eval()
    with torch.no_grad():
        k = net.forward_argmax_decode(image)
        k = k.data.cpu().numpy()
        k = tokenizer.predict_to_inchi(k)
        text.extend(k)

    print('')
    return text


initial_checkpoint = '/Users/zhenyuanzhang/Desktop/udemy/AWS_Lambda_and_Serverless/IMG2InChi/348000_model.pth'#None #

is_norm_ichi = False #True

## setup  ----------------------------------------
mode = 'local'

## dataset ------------------------------------
tokenizer = util.load_tokenizer()
#net = Net().cuda()
net = Net()
net.load_state_dict(torch.load(initial_checkpoint, map_location=torch.device('cpu'))['state_dict'], strict=True)
#image_file = '/Users/zhenyuanzhang/Desktop/udemy/AWS_Lambda_and_Serverless/IMG2InChi/data/d/1dd0ac1d7607.png'

#---
# predict = do_predict(net, tokenizer, image_file)

# print(predict)

@app.get('/')
def my_function(image_file:str):
  pred = do_predict(net, tokenizer, image_file)
  return JSONResponse({"prediction":pred})

if __name__=="__main__":
  uvicorn.run(app,host="0.0.0.0",port=9000)   #http://0.0.0.0:9000/docs#/