import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
from os import getenv
from openai import OpenAI
import pathlib
import textwrap
import google.generativeai as genai
import time
import requests
import ast
import json
from sklearn.metrics.pairwise import cosine_similarity
import re

load_dotenv("../../.env",override=True)
GOOGLE_API_KEY = getenv("GEMINI_API_KEY")
OPENAI_API_KEY = getenv("OPENAI_API_KEY")



genai.configure(api_key=GOOGLE_API_KEY)
class gemini():
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    def request(self,prompt):
        url = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent'
        headers = {
            'Content-Type': 'application/json',
        }
        data = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ]
        }
        params = {
            'key': GOOGLE_API_KEY
        }
        
        response = requests.post(url, headers=headers, json=data, params=params)
        return json.loads(response.text)

    def ask(self,prompt):
        #response = self.model.generate_content(prompt)
        response = self.request(prompt)
        if response["candidates"][0]["finishReason"] == 'SAFETY': return "N/A"
        response = response["candidates"][0]["content"]["parts"][0]["text"]
        return response



class chatGPT():
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        
    # def ask(self, q):
    #     stream = self.client.chat.completions.create(
    #         model="gpt-4",
    #         messages=[{"role": "user", "content": q}],
    #         stream=True,
    #         temperature=0
    #     )
    #     response = ""
    #     for chunk in stream:
    #         if chunk.choices[0].delta.content is not None:
    #             response += chunk.choices[0].delta.content

    #     self.response = response
    #     return response

    def get_embedding(self,text, model="text-embedding-3-large"):
       text = text.replace("\n", " ")
       return self.client.embeddings.create(input = [text], model=model).data[0].embedding






