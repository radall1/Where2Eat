import os
import pickle
import openai
import tiktoken
import pandas as pd
import numpy as np

from openai import OpenAI
from scipy import spatial
from ast import literal_eval
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

GPT_MODEL = 'gpt-3.5-turbo'

def read_secret_file(filename):
    filepath = os.path.join("/etc/secrets", filename)
    with open(filepath, "r") as file:
        return file.read().strip()

API_KEY = read_secret_file("openai_api_key.txt")

client = OpenAI(api_key=API_KEY) 
model = "text-embedding-3-small"

mini_dataset_files = ["datasets/reviews_with_embeddings_1.csv", "datasets/reviews_with_embeddings_4.csv"]

def join_datasets(input_files):
    return pd.concat([pd.read_csv(file) for file in input_files], ignore_index=True)
    
df = join_datasets(mini_dataset_files)
df["embedding"] = df.embedding.apply(literal_eval).apply(np.array)

def get_embedding(text, model=model):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def strings_ranked_by_relatedness(query, df, relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y), top_n = 5):
    query_embedding = get_embedding(query)
    strings_and_relatednesses = [
        (row["combined"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]
    
def num_tokens(text, model = GPT_MODEL) -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def query_message(query, df, model, token_budget):
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = 'Let the user know (1) that the following are the top 3 restaurants recommended to them based on what they have asked for, (2) why those restaurants are a good fit for them and (3) back your justifications with quotes from the reviews. Write in paragraph, conversational style. If the answer cannot be found in the reviews, write "I could not find an answer."'
    question = f"\n\What they asked for: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\n{string}\n'
        if (num_tokens(message + next_article + question, model=model) > token_budget):
            break
        else:
            message += next_article
    return message + question

def ask(query, df = df, model = GPT_MODEL, token_budget = 4096 - 500, print_message = False):
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions."},
        {"role": "user", "content": message},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response.choices[0].message.content
    return response_message

@app.route('/', methods=['POST'])
def handle_input():
    user_input = request.json.get('input', '')
    output = ask(user_input)
    return jsonify({'output': output})
    
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False)
