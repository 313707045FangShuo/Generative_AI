import zipfile
import os
import getpass
import pandas as pd
import random
import time
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# unzip dataset
zip_file = "hw-1-prompt-engineering.zip"

with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall("HW_LLM/HW1_prompt")

# os get genmini api key
file_path = "/mnt/sda1/shuof/HW_LLM/HW1_prompt/genmini_api.txt"  # api_key

with open(file_path, "r", encoding="utf-8") as file:
    google_api_key = file.read()

os.environ["GOOGLE_API_KEY"] = google_api_key

# prevent requset to genmini exceede the free limit
MAX_REQUESTS_PER_MINUTE = 10
REQUEST_INTERVAL = 60 / MAX_REQUESTS_PER_MINUTE

MAX_TOKENS_PER_MINUTE = 20000
tokens_used = 0
start_time = time.time()

sample_file_path = "HW_LLM/HW1_prompt/dataset/mmlu_sample.csv"
df_sample = pd.read_csv(sample_file_path)

submit_file_path = "HW_LLM/HW1_prompt/dataset/mmlu_submit.csv"
df_submit = pd.read_csv(submit_file_path)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

n_shot = 2

results = []

for i, row in df_submit.iterrows():
    request_start_time = time.time()

    task = row["task"]
    input_question = row["input"]
    options = {"A": row["A"], "B": row["B"], "C": row["C"], "D": row["D"]}

    # get the same task with this question from sample.csv as few-shot sample
    few_shot_samples = df_sample[df_sample["task"] == task].sample(n=min(n_shot, len(df_sample[df_sample["task"] == task])), random_state=42)

    # few-shot
    prompt = "Task: Answer multiple-choice questions by selecting the correct option (A, B, C, or D). Do not explain, just output the correct letter.\n\n"

    for _, example in few_shot_samples.iterrows():
        prompt += f"Example:\nQuestion: {example['input']}\nA) {example['A']}\nB) {example['B']}\nC) {example['C']}\nD) {example['D']}\nAnswer: {example['target']}\n\n"

    prompt += f"\nNow answer the following question:\n\nQuestion: {input_question}\nA) {options['A']}\nB) {options['B']}\nC) {options['C']}\nD) {options['D']}\nAnswer:"
    
    print(f"Processing question {i+1}")

    # estimated token amount
    estimated_tokens = len(prompt.split()) + 100

    # check TPM
    if tokens_used + estimated_tokens > MAX_TOKENS_PER_MINUTE:
        elapsed_time = time.time() - start_time
        wait_time = max(0, 60 - elapsed_time)  # 間隔一分鐘
        print(f"Token limit reached, waiting {wait_time:.2f} seconds...")
        time.sleep(wait_time)
        tokens_used = 0  # reset Token
        start_time = time.time()

    # LLM
    message = HumanMessage(content=[{"type": "text", "text": prompt}])
    response = llm.invoke([message])
    # get result
    predicted_answer = response.content.strip().upper()

    # store result
    print(f'"ID": {row["Unnamed: 0"]}, "target": {predicted_answer}')
    results.append({"ID": row["Unnamed: 0"], "target": predicted_answer})

    # token++
    tokens_used += estimated_tokens

    # time++
    request_elapsed_time = time.time() - request_start_time
    time.sleep(max(0, REQUEST_INTERVAL - request_elapsed_time))

df_results = pd.DataFrame(results)
output_file_path = "HW_LLM/HW1_prompt/dataset/mmlu_predictions.csv"
df_results.to_csv(output_file_path, index=False)

print(f"Results saved to {output_file_path}")