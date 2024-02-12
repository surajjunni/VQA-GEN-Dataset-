import pandas as pd
from transformers import MarianMTModel, MarianTokenizer

def backtranslate(sentence, model, tokenizer, source_lang="en", target_lang="fr"):
    # Translate from source to target language
    translated = model.generate(**tokenizer(sentence, return_tensors="pt", padding=True, truncation=True), max_length=50)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    
    # Translate back to source language
    backtranslated = model.generate(**tokenizer(translated_text, return_tensors="pt", padding=True, truncation=True), max_length=50)
    backtranslated_text = tokenizer.decode(backtranslated[0], skip_special_tokens=True)

    return backtranslated_text

# Load the Marian MT model and tokenizer
model_name = "Helsinki-NLP/opus-mt-en-fr"
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Read the CSV file with questions
csv_file_path = "path/to/your/csvfile.csv"
df = pd.read_csv(csv_file_path)

# Apply backtranslation to each question in the DataFrame
df["Backtranslated_Question"] = df["Question"].apply(lambda x: backtranslate(x, model, tokenizer))

# Save the DataFrame with backtranslated questions to a new CSV file
output_csv_path = "path/to/your/output/csvfile_backtranslated.csv"
df.to_csv(output_csv_path, index=False)

print(f"Backtranslated questions saved to {output_csv_path}")
