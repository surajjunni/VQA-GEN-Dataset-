import openai
import re

openai.api_key = "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"
import pandas as pd

# Load the input CSV file
input_data = pd.read_csv('batch_26.csv')

# Define a function to generate declarative sentences from questions and answers
def generate_sentence(row):
    try:
        question = row['question']
        message=[
{"role": "system", "content": "You are a helpful assistant."},
{"role": "user", "content": f"Q:'{question}' Personas:Indian English(critic,professor,elementary school student,college student,high school student),American English(critic,professor,elementary school student,college student,high school student),British English(critic,professor,elementary school student,college student,high school student)"},
{"role": "user", "content": f"As the context of the question in this case is dependent on the  image, create various question styles according to the specified personas, and the styles should correspond to the question and shouldn't discuss how each persona relates to it. instead, I only need to know how each character asks the question.. The output format should be: \n{{'Indian English':{{'critic': style1, 'professor':style2, 'elementary school student':style3, 'college student':style4, 'high school student':style5}}, 'American English':{{'critic': style1, 'professor':style2, 'elementary school student':style3, 'college student':style4, 'high school student':style5}}, 'British English':{{'critic': style1, 'professor':style2, 'elementary school student':style3, 'college student':style4, 'high school student':style5}}}}"}
]

        # Generate a declarative sentence using the OpenAI API
        response=openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=message)

        # Extract the generated sentence from the API response
        sentence = response.choices[0]["message"]["content"]

        return sentence
    except Exception as e:
        print(f"Error while generating sentence for question '{question}': {e}")
        return None

# Apply the function to each row of the input data to generate declarative sentences
output_data = pd.DataFrame()  # Create an empty DataFrame to store the output
for index, row in input_data.iterrows():
    sentence = generate_sentence(row)
    if sentence is not None:
        output_data = output_data.append({'question': row['question'], 'sentence': sentence}, ignore_index=True)
    else:
        print(f"Skipping question '{row['question']}' due to an error.")
    output_data.to_csv('output_data27.csv', index=False)  # Save the output CSV file

