import os
import pandas as pd
import torch
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering, AdamW, get_linear_schedule_with_warmup

# Set up the device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load the CSV file with questions and answers
csv_file = 'merged_table.csv'
df = pd.read_csv(csv_file)

# Initialize the VILT processor and model
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa").to(device)
print(model)
# Set up the optimizer and learning rate schieduler
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=1000)

# Train the model on your data
root_dir = '/home/ubuntu/suraj/pytorch-AdaIN'
image_dirs = ['domain1', 'domain2', 'domain3', 'domain4', 'domain5', 'domain6']

for epoch in range(60):
    print(f"Epoch {epoch + 1}")
    model.train()
    for image_dir in image_dirs:
        image_dir_path = os.path.join(root_dir, image_dir)
        for filename in os.listdir(image_dir_path):
            if filename.endswith('.jpg'):
                # Extract the image ID from the filename
                image_id = filename.split('_')[-1].split('.')[0]

                # Find the corresponding rows in the CSV file
                rows = df.loc[df['image_id'] == int(image_id)]

                # Load the image file
                image_path = os.path.join(image_dir_path, filename)
                image = Image.open(image_path)

                # Process the question/answer data
                for _, row in rows.iterrows():
                    question = row['question']
                    label = row['answer']
                    encoding = processor(text=question, images=image, return_tensors="pt").to(device)
                    #label = torch.tensor(processor.tokenizer.convert_tokens_to_ids(answer), device=device)
                    print(label)                   
                    # Forward pass and compute loss
                    #outputs = model(**encoding)
                    #print(label.shape)
                    #print(model.config.label2id)
                    #label=model.config.label2id[label]
                    tensor = torch.zeros(3129)
                    # Set the value at index 123 to 1
                    try:
                        label=model.config.label2id[label]
                        tensor[label] = 1
                    except:
                        tensor[545]=1
                    tensor=torch.unsqueeze(tensor, 0)
                    #label = torch.tensor(label, device=device)
                    #print(label)
                    #print(tensor.shape)
                    outputs=model(**encoding,labels=tensor.to(device))
                    #predicted_labels = outputs.argmax(dim=1)
                    loss = outputs.loss
                    # Backward pass and update model weights
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

    # Test the model on VQA v2 validation dataset
    print("Testing the model on VQA v2 validation dataset")
    model.eval()
    image_dir_path = "~/suraj/val2014"
    count=0
    total=0
    for filename in os.listdir(image_dir_path):
        if filename.endswith('.jpg'):
            # Extract the image ID from the filename
            image_id = filename.split('_')[-1].split('.')[0]

            # Find the corresponding rows in the CSV file
            rows = df.loc[df['image_id'] == int(image_id)]

            # Load the image file
            image_path = os.path.join(image_dir_path, filename)
            image = Image.open(image_path)

            # Apply the transformer to the image
            #transformed_image = vilt.transform_image(image)

            # Process the question/answer data
            for _, row in rows.iterrows():
                question = row['question']
                answer = row['answer']
                encoding = processor(text=question, images=image, return_tensors="pt")
                outputs = model(**encoding)
                logits = outputs.logits
                idx = logits.argmax(-1).item()
                predicted_answer = model.config.id2label[idx]
                print(f'Image ID: {image_id}, Question: {question}, Answer: {answer}, Predicted answer: {predicted_answer}')

                # Check if the predicted answer matches the actual answer
                if predicted_answer == answer:
                    count += 1
                total += 1
    print("Accuracy",count/total)         
