import json, io, subprocess, base64
from time import time

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
import requests 
#import cv2


url = 'https://dev263135.service-now.com/x_146833_redhatpre.brakesetpads.jpg'
testimage = 'aniltestimage.jpg'

response = requests.get(url)
if response.status_code == 200:
    with open(testimage, 'wb') as file:
        file.write(response.content)
    print('File downloaded successfully')
else:
    print('Failed to download file')

api_key = "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" #FIX ME 

#Setup VLM NIM Urls 
neva_api_url = "https://ai.api.nvidia.com/v1/vlm/nvidia/neva-22b"

def process_image(image):
    """ Resize image, encode as jpeg to shrink size then convert to b64 for upload """
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")
        
    image = image.resize((336,336)) #Resize or center crop and padding to be square are common approaches 
    buf = io.BytesIO() #temporary buffer to save processed image 
    image.save(buf, format="JPEG") #save as jpeg to reduce size
    image = buf.getvalue()
    image_b64 = base64.b64encode(image).decode() #convert to b64 string
    assert len(image_b64) < 180_000, "Image to large to upload." #ensure image is small enough
    return image_b64
headers = {
  "Authorization": f"Bearer {api_key}",
  "Accept": "application/json"
}

image_b64 = process_image(testimage) #put the filepath to your own image
payload = {
  "messages": [
    {
      "role": "user",
      #"content": f'Describe what you see in this image. <img src="data:image/jpeg;base64,{image_b64}" />'
        "content": f'You are seeing brake pads in the image can you tell if they are having good grip on them <img src="data:image/jpeg;base64,{image_b64}" />'
    }
  ],
  "max_tokens": 1024, #TEST
  "temperature": 0.20, #TEST
  "top_p": 0.70, #TEST
  "seed": 0, #TEST
  "stream": False
}

response = requests.post(neva_api_url, headers=headers, json=payload)
response = response.json()

class VLM:
    def __init__(self, url, api_key):
        """ Provide NIM API URL and an API key"""
        self.api_key = api_key
        self.url = url 
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}

    def _encode_image(self, image):
        """ Resize image, encode as jpeg to shrink size then convert to b64 for upload """

        if isinstance(image, str): #file path
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image): #pil image 
            image = image.convert("RGB")
       # elif isinstance(image, np.ndarray): #cv2 / np array image 
        #    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
         #   image = Image.fromarray(image)
        else:
            print(f"Unsupported image input: {type(image)}")
            return None 
            
        image = image.resize((336,336))
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        image = buf.getvalue()
        image_b64 = base64.b64encode(image).decode()
        assert len(image_b64) < 180_000, "Image too large to upload."
        return image_b64

    def __call__(self, prompt, image):
        """ Call VLM object with the prompt and path to image """ 
        image_b64 = self._encode_image(image)

        #For simplicity, the image will be appended to the end of the prompt. 
        payload = {
              "messages": [
                {
                  "role": "user",
                  "content": f'{prompt} Here is the image: <img src="data:image/jpeg;base64,{image_b64}" />'
                }
              ],
              "max_tokens": 128,
              "temperature": 0.20,
              "top_p": 0.70,
              "stream": False
        }

        response = requests.post(self.url, headers=headers, json=payload)
        response = response.json()
        reply = response["choices"][0]["message"]["content"]
        return reply, response #return reply and the full response
    
    #Create a VLM object for each supported model 
neva = VLM(neva_api_url, api_key)

custom_prompt = "f'You are seeing brake pads in the image can you tell if they are having good grip on them" #CHANGE ME
image_path = testimage #CHANGE ME

#NEVA
start_time = time()
response, _ = neva(custom_prompt, image_path)
print(f"Neva Response: {response}")
print(f"Neva Time: {time() - start_time} \n")


def predict(single_test_text):
    image_path = testimage
    response, _ = neva(single_test_text,image_path)
    single_predicted_label = response
    return {'prediction': single_predicted_label}

print(predict('what do you see in the image'))
