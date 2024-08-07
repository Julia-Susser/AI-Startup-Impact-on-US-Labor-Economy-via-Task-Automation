import nest_asyncio
import aiohttp
import asyncio
import nest_asyncio
import re
import time
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
from llms import chatGPT
from llms import gemini


x_chat = chatGPT()
x_gemini = gemini()

# Apply nest_asyncio
nest_asyncio.apply()


class prompting():
    def __init__(self):
        self.results_df = pd.DataFrame(columns=["organization name", "value"])

    def set_current_results_df(self, results_df):
        self.results_df = results_df

    async def iterate(self, df, prompt_template, args, value, batch_size=10, start=0, end=False):
        if end == False:
            end = len(df)
        self.results_df = pd.DataFrame(columns=["organization name", value])
        if value in list(df.columns):
            if start != 0:
                self.results_df = pd.concat([df[["organization name", value]].iloc[:start], self.results_df], axis=0)
            df = df.drop(columns=[value])

        batch_prompts = []
        batch_indices = []

        for i, row in list(df.iterrows())[start:end]:
            name = row['organization name']
            website = row['website']
            prompt = prompt_template

            for arg in args:
                prompt = prompt.replace(f"${arg[0]}", row[arg[1]])

            batch_prompts.append((i, name, prompt))
            batch_indices.append(i)

            if len(batch_prompts) >= batch_size:
                await self.process_batch(batch_prompts)
                batch_prompts = []

        if batch_prompts:
            await self.process_batch(batch_prompts)
            self.results_df.to_csv("../output/current_results_df_prompting.csv")

        df = df.merge(self.results_df, on='organization name', how='left')
        return df

    async def process_batch(self, batch_prompts):
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i, name, prompt in batch_prompts:
                tasks.append(self.fetch_result(session, i, name, prompt))
            await asyncio.gather(*tasks)

    async def fetch_result(self, session, i, name, prompt):
        failure_count = 0
        while True:
            try:
                print(f"******************************\nProcessing {i}: {name}")
                result = await x_gemini.ask(session, prompt)
                if result == "N/A": break  # explicit material

                text = re.sub(r"#|#\s+|_|\*", "", result).strip()

                self.results_df.loc[i] = [name, text]

                    
                break
            except Exception as e:
                print(failure_count)
                failure_count += 1
                if failure_count > 10:
                    break
                print(f"Error processing {i}, {name}: {e}")
                await asyncio.sleep(20)

prompting_class = prompting()
