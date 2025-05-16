import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

# load test
df = pd.read_csv("test.csv")#.sample(20, random_state=42)  # reduce for faster runs
texts = df["full_text"].tolist()
true_labels = df["score"].apply(lambda x: 1 if x > 0 else 0).tolist()

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

def classify_with_prompt(text):
    prompt = f"Given Tweet: '{text[0:50]}'\n - Tell me the sentiment (positive/negative):"
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=inputs.shape[1] + 20,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_text = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip().lower()
    print(f"Generated Text: '{generated_text}'")

    
    if "positive" in generated_text or "good" in generated_text or "happy" in generated_text or "pos" in generated_text:
        return 1
    elif "negative" in generated_text or "bad" in generated_text or "sad" in generated_text or "neg" in generated_text:
        return 0
    return None 

#calassification
preds = []
for text in tqdm(texts, desc="Classifying tweets"):
    pred = classify_with_prompt(text)
    preds.append(pred)

#Filter results
valid_indices = [i for i, p in enumerate(preds) if p is not None]
filtered_preds = [preds[i] for i in valid_indices]
filtered_truth = [true_labels[i] for i in valid_indices]

#Evaluate
print("GPT-2 Prompt-Based Evaluation")
print(classification_report(filtered_truth, filtered_preds))
print("Accuracy:", accuracy_score(filtered_truth, filtered_preds))

results_df = pd.DataFrame({
    "tweet": texts,
    "true_label": true_labels,
    "predicted_label": preds
})

#filter out invalid preds then save
results_df = results_df[results_df["predicted_label"].notnull()]
results_df.to_csv("gpt_sentiment_results.csv", index=False)
print("Results saved to gpt_sentiment_results.csv")
