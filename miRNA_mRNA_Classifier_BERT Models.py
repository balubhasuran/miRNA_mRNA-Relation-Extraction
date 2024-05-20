#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import random
import numpy as np
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import torch
import spacy
from spacy import displacy
nlp = spacy.blank('en')
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer



# In[2]:


get_ipython().system('pip install autogluon')


# In[4]:


get_ipython().system('pip install spacy')


# In[2]:


from transformers import get_linear_schedule_with_warmup


# In[3]:


import transformers
print(transformers.__version__)


# In[222]:


#os.mkdir('models')
df=pd.read_csv('D:\\e Health Lab projects\\Question_Answering\\Classification\\QA_Classification.csv',engine='python',encoding = 'latin',keep_default_na=False)
print(df)


# In[223]:


df['Tags'].value_counts()


# In[224]:


evidence_labels = df.Tags.unique()


# In[225]:


evidence_dict = {}
for index, evidence_labels in enumerate(evidence_labels):
    evidence_dict[evidence_labels] = index
evidence_dict


# In[226]:


df['Tags Mapped'] = df.Tags.map(evidence_dict)
print(df)


# In[227]:


X_train, X_val, y_train, y_val = train_test_split(df.index.values,\
                                                  df['Tags Mapped'].values,\
                                                  test_size=0.20,\
                                                  random_state=42,\
                                                  stratify=df['Tags Mapped'].values)

df['data_type'] = ['not_set']*df.shape[0]
df.loc[X_train, 'data_type'] = 'train'
df.loc[X_val, 'data_type'] = 'val'


# In[229]:


#tokenizer = BertTokenizer.from_pretrained('D:\\LLM Models\\BERT Models\\models\\biobert-base-cased-v1.2\\', do_lower_case=True,truncation=True)
#tokenizer = BertTokenizer.from_pretrained('D:\\LLM Models\\BERT Models\\models\\allenaiscibert_scivocab_uncased\\', do_lower_case=True,truncation=True)
tokenizer = BertTokenizer.from_pretrained('D:\\LLM Models\\BERT Models\\models\\emilyalsentzerBio_ClinicalBERT\\', do_lower_case=True,truncation=True)
#tokenizer = BertTokenizer.from_pretrained('D:\\LLM Models\\BERT Models\\models\\medicalaiClinicalBERT\\', do_lower_case=True,truncation=True)
#tokenizer = BertTokenizer.from_pretrained('D:\\LLM Models\\BERT Models\\models\\microsoftBiomedNLP-PubMedBERT-base-uncased-abstract-fulltext\\', do_lower_case=True,truncation=True)


# In[230]:


token_lens = []
for txt in df.Sentences:
  tokens = tokenizer.encode(txt, max_length=512)
  token_lens.append(len(tokens))


# In[231]:


import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
sns.distplot(token_lens)
plt.xlim([0, 256]);
plt.xlabel('Token count');


# In[156]:


encoded_data_train = tokenizer.batch_encode_plus(
    df[df.data_type=='train'].Sentences.values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    truncation=True,
    padding='longest', 
    max_length=256, 
    return_tensors='pt'
)


# In[157]:


encoded_data_val = tokenizer.batch_encode_plus(
    df[df.data_type=='val'].Sentences.values, 
    add_special_tokens=True, 
    truncation=True,
    return_attention_mask=True, 
    padding='longest', 
    max_length=256, 
    return_tensors='pt'
)


# In[158]:


input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
tags_train = torch.tensor(df[df.data_type=='train']['Tags Mapped'].values)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
tags_val = torch.tensor(df[df.data_type=='val']['Tags Mapped'].values)

dataset_train = TensorDataset(input_ids_train, attention_masks_train, tags_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, tags_val)


# In[159]:


print(len(dataset_train), len(dataset_val))


# In[160]:


#model = BertForSequenceClassification.from_pretrained("D:\\LLM Models\\BERT Models\\models\\biobert-base-cased-v1.2", num_labels=2,output_attentions=False,output_hidden_states=False)
#model = BertForSequenceClassification.from_pretrained("D:\\LLM Models\\BERT Models\\models\\allenaiscibert_scivocab_uncased", num_labels=2,output_attentions=False,output_hidden_states=False)
model = BertForSequenceClassification.from_pretrained("D:\\LLM Models\\BERT Models\\models\\emilyalsentzerBio_ClinicalBERT", num_labels=2,output_attentions=False,output_hidden_states=False)
#model = BertForSequenceClassification.from_pretrained("D:\\LLM Models\\BERT Models\\models\\medicalaiClinicalBERT", num_labels=2,output_attentions=False,output_hidden_states=False)
#model = BertForSequenceClassification.from_pretrained("D:\\LLM Models\\BERT Models\\models\\microsoftBiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", num_labels=2,output_attentions=False,output_hidden_states=False)

batch_size = 16 #Defining Batch Size on which model has to be trained
# Converting tensors Dataset to dataloaders so that model can be trained
dataloader_train = DataLoader(dataset_train,sampler=RandomSampler(dataset_train), batch_size=batch_size) 
dataloader_validation = DataLoader(dataset_val, sampler=SequentialSampler(dataset_val), batch_size=batch_size)
optimizer = AdamW(model.parameters(),lr=8e-7, eps=1e-8) # Initializing Adam Weight Decay Optimizer with its parameters 
epochs = 30
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,num_training_steps=len(dataloader_train)*epochs)


# In[161]:


def f1_score_func(preds, tags):
    preds_flat = np.argmax(preds, axis=1).flatten()
    tags_flat = tags.flatten()
    return f1_score(tags_flat, preds_flat, average='weighted')


# In[162]:


def accuracy_per_class(preds, tags):
    evidence_dict_inverse = {v: k for k, v in evidence_dict.items()}
    
    preds_flat = np.argmax(preds, axis=1).flatten()
    tags_flat = tags.flatten()
    for tag in np.unique(tags_flat):
        y_preds = preds_flat[tags_flat==tag]
        y_true = tags_flat[tags_flat==tag]
        print(f'Question Class: {evidence_dict_inverse[tag]}')
        print(f'Accuracy: {len(y_preds[y_preds==tag])}/{len(y_true)}\n')


# In[163]:


seed_val = 123
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
model.to(device)
print(device)


# In[164]:


def evaluate(dataloader_val):
    model.eval()
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals


# In[165]:


loss_train_avg_list = []
val_loss_list = []
val_f1_list = []


# In[97]:


#BioBERT
for epoch in tqdm(range(1, epochs+1)):
    
    model.train()
    
    loss_train_total = 0

    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    for batch in progress_bar:

        model.zero_grad()
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }       

        outputs = model(**inputs)
        
        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        model.to('cuda')
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
         
    torch.save(model.state_dict(), f'models/finetuned_BioBERT_epoch_{epoch}.model')
        
    tqdm.write(f'\nEpoch {epoch}')
    
    loss_train_avg = loss_train_total/len(dataloader_train)            
    tqdm.write(f'Training loss: {loss_train_avg}')
    loss_train_avg_list.append(loss_train_avg)
    
    val_loss, predictions, true_vals = evaluate(dataloader_validation)
    val_f1 = f1_score_func(predictions, true_vals)
    tqdm.write(f'Validation loss: {val_loss}')
    val_loss_list.append(val_loss)
    tqdm.write(f'F1 validation Score (Weighted): {val_f1}')
    val_f1_list.append(val_f1)


# In[98]:


model.load_state_dict(torch.load('models/finetuned_BioBERT_epoch_25.model', map_location=torch.device('cuda')))
sns.set(style='darkgrid')
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12,6)
plt.plot(loss_train_avg_list, 'b-o')
plt.title("Training loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


# In[99]:


sns.set(style='darkgrid')
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12,6)
plt.plot(val_loss_list, 'b-o')
plt.title("Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


# In[100]:


sns.set(style='darkgrid')
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12,6)
plt.plot(val_f1_list, 'b-o')
plt.title("Validation F1 Score")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


# In[101]:


_, predictions, true_vals = evaluate(dataloader_validation)
accuracy_per_class(predictions, true_vals)


# In[166]:


#ClinicalBERT
for epoch in tqdm(range(1, epochs+1)):
    
    model.train()
    
    loss_train_total = 0

    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    for batch in progress_bar:

        model.zero_grad()
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }       

        outputs = model(**inputs)
        
        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        model.to('cuda')
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
         
    torch.save(model.state_dict(), f'models/finetuned_ClinicalBERT_epoch_{epoch}.model')
        
    tqdm.write(f'\nEpoch {epoch}')
    
    loss_train_avg = loss_train_total/len(dataloader_train)            
    tqdm.write(f'Training loss: {loss_train_avg}')
    loss_train_avg_list.append(loss_train_avg)
    
    val_loss, predictions, true_vals = evaluate(dataloader_validation)
    val_f1 = f1_score_func(predictions, true_vals)
    tqdm.write(f'Validation loss: {val_loss}')
    val_loss_list.append(val_loss)
    tqdm.write(f'F1 validation Score (Weighted): {val_f1}')
    val_f1_list.append(val_f1)


# In[167]:


model.load_state_dict(torch.load('models/finetuned_ClinicalBERT_epoch_30.model', map_location=torch.device('cuda')))
sns.set(style='darkgrid')
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12,6)
plt.plot(loss_train_avg_list, 'b-o')
plt.title("Training loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


# In[168]:


sns.set(style='darkgrid')
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12,6)
plt.plot(val_loss_list, 'b-o')
plt.title("Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


# In[169]:


sns.set(style='darkgrid')
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12,6)
plt.plot(val_f1_list, 'b-o')
plt.title("Validation F1 Score")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


# In[170]:


_, predictions, true_vals = evaluate(dataloader_validation)
accuracy_per_class(predictions, true_vals)


# In[171]:


reloaded_results = model.load_state_dict(torch.load('models/finetuned_ClinicalBERT_epoch_30.model', map_location=torch.device('cuda')))


# In[187]:


seed_val = 123
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
model.to(device)
print(device)


# In[248]:


model = BertForSequenceClassification.from_pretrained("D:\\LLM Models\\BERT Models\\models\\emilyalsentzerBio_ClinicalBERT", num_labels=2,output_attentions=False,output_hidden_states=False)
tokenizer = BertTokenizer.from_pretrained('D:\\LLM Models\\BERT Models\\models\\emilyalsentzerBio_ClinicalBERT\\', do_lower_case=True,truncation=True)


# In[256]:


def predict(review_text):
    encoded_review = tokenizer.encode_plus(
    review_text,
    max_length=256,
    add_special_tokens=True,
    return_token_type_ids=False,
    pad_to_max_length=True,
    return_attention_mask=True,
    return_tensors='pt',
    )

    input_ids = encoded_review['input_ids']
    attention_mask = encoded_review['attention_mask']
    output = model(input_ids, attention_mask)
    _, prediction = torch.max(output[0], dim=1)
    print(f'Lab Question: {review_text}')
    print(f'Label  : {class_names[prediction]}')
    return class_names[prediction]


# In[257]:


class_names = ['positive','negative']


# In[259]:


df0 = pd.read_csv('D:\\e Health Lab projects\\Question_Answering\\Classification\\Yahoo answer mapped labs from LabGenie.csv',engine='python',encoding = 'latin',keep_default_na=False)
df=df0[['question']]


# In[262]:


df["Tags"] = df.apply(lambda l: predict(l.question), axis=1)


# In[263]:


df.to_csv('D:\\e Health Lab projects\\Question_Answering\\Classification\\result.csv')


# In[127]:


#SciBERT
for epoch in tqdm(range(1, epochs+1)):
    
    model.train()
    
    loss_train_total = 0

    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    for batch in progress_bar:

        model.zero_grad()
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }       

        outputs = model(**inputs)
        
        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        model.to('cuda')
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
         
    torch.save(model.state_dict(), f'models/finetuned_SciBERT_epoch_{epoch}.model')
        
    tqdm.write(f'\nEpoch {epoch}')
    
    loss_train_avg = loss_train_total/len(dataloader_train)            
    tqdm.write(f'Training loss: {loss_train_avg}')
    loss_train_avg_list.append(loss_train_avg)
    
    val_loss, predictions, true_vals = evaluate(dataloader_validation)
    val_f1 = f1_score_func(predictions, true_vals)
    tqdm.write(f'Validation loss: {val_loss}')
    val_loss_list.append(val_loss)
    tqdm.write(f'F1 validation Score (Weighted): {val_f1}')
    val_f1_list.append(val_f1)


# In[128]:


model.load_state_dict(torch.load('models/finetuned_SciBERT_epoch_25.model', map_location=torch.device('cuda')))
sns.set(style='darkgrid')
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12,6)
plt.plot(loss_train_avg_list, 'b-o')
plt.title("Training loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


# In[129]:


sns.set(style='darkgrid')
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12,6)
plt.plot(val_loss_list, 'b-o')
plt.title("Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


# In[130]:


sns.set(style='darkgrid')
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12,6)
plt.plot(val_f1_list, 'b-o')
plt.title("Validation F1 Score")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


# In[131]:


_, predictions, true_vals = evaluate(dataloader_validation)
accuracy_per_class(predictions, true_vals)


# In[143]:


#PubMedBERT
for epoch in tqdm(range(1, epochs+1)):
    
    model.train()
    
    loss_train_total = 0

    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    for batch in progress_bar:

        model.zero_grad()
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }       

        outputs = model(**inputs)
        
        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        model.to('cuda')
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
         
    torch.save(model.state_dict(), f'models/finetuned_PubMedBERT_epoch_{epoch}.model')
        
    tqdm.write(f'\nEpoch {epoch}')
    
    loss_train_avg = loss_train_total/len(dataloader_train)            
    tqdm.write(f'Training loss: {loss_train_avg}')
    loss_train_avg_list.append(loss_train_avg)
    
    val_loss, predictions, true_vals = evaluate(dataloader_validation)
    val_f1 = f1_score_func(predictions, true_vals)
    tqdm.write(f'Validation loss: {val_loss}')
    val_loss_list.append(val_loss)
    tqdm.write(f'F1 validation Score (Weighted): {val_f1}')
    val_f1_list.append(val_f1)


# In[144]:


model.load_state_dict(torch.load('models/finetuned_PubMedBERT_epoch_1.model', map_location=torch.device('cuda')))
sns.set(style='darkgrid')
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12,6)
plt.plot(loss_train_avg_list, 'b-o')
plt.title("Training loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


# In[145]:


sns.set(style='darkgrid')
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12,6)
plt.plot(val_loss_list, 'b-o')
plt.title("Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


# In[146]:


sns.set(style='darkgrid')
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12,6)
plt.plot(val_f1_list, 'b-o')
plt.title("Validation F1 Score")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


# In[147]:


_, predictions, true_vals = evaluate(dataloader_validation)
accuracy_per_class(predictions, true_vals)


# In[ ]:




