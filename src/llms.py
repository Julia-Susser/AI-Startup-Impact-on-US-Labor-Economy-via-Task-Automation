import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
from os import getenv
from openai import OpenAI
import pathlib
import textwrap
# import google.generativeai as genai
import time
import requests
import ast
import json
from sklearn.metrics.pairwise import cosine_similarity
import re


# 
# genai.configure(api_key=GOOGLE_API_KEY)

load_dotenv("../.env",override=True)
GOOGLE_API_KEY = getenv("GEMINI_API_KEY")
OPENAI_API_KEY = getenv("OPENAI_API_KEY")



class gemini():
    async def request(self, session, prompt):
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
        
        async with session.post(url, headers=headers, json=data, params=params) as response:
            return await response.json()

    async def ask(self, session, prompt):
        response = await self.request(session, prompt)
        if response["candidates"][0]["finishReason"] == 'SAFETY': return "N/A"
        return response["candidates"][0]["content"]["parts"][0]["text"]

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


    def get_batch_embeddings(self, texts, model="text-embedding-3-large"):
        texts = [text.replace("\n", " ") for text in texts]
        response = self.client.embeddings.create(input=texts, model=model)
        return [item.embedding for item in response.data]
    

    def run_batch_embeddings(self, df, col, batch_size=200, model="text-embedding-3-large"):
        results = []
        batch_texts = []
        indices = []
        start = 0
        num = 0
        for i, text in df[col].items():
            batch_texts.append(text)
            indices.append(i)
            
            # When we reach the batch size, process the batch
            if len(batch_texts) >= batch_size:
                print(f"Processing batch {start} to {num}")

                # Get embeddings for the batch of texts
                batch_embeddings = self.get_batch_embeddings(batch_texts, model=model)

                # Append results
                for idx, embedding in zip(indices, batch_embeddings):
                    results.append(embedding)

                # Clear the batch lists
                batch_texts = []
                indices = []
                start = num+1
            num = num+1
        # Process any remaining items in the batch
        if batch_texts:
            print(f"Processing final batch {len(df) - len(batch_texts)} to {len(df)}")
            batch_embeddings = self.get_batch_embeddings(batch_texts, model=model)
            for idx, embedding in zip(indices, batch_embeddings):
                results.append(embedding)

        # Create a DataFrame from the results
        df[f'{col}_embedding'] = results

        return df




