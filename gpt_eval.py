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
    prompt = f"Tweet: {text}\nSentiment (Positive or Negative):"
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones_like(inputs)
    outputs = model.generate(inputs, 
                             attention_mask=attention_mask, 
                             max_length=inputs.shape[1] + 5, 
                             do_sample=False, 
                             pad_token_id=tokenizer.eos_token_id)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Positive" in output_text:
        return 1
    elif "Negative" in output_text:
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
