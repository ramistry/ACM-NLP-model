# Urgency Classifier

## 1. Background and Motivation
Let's say there are two messages for help – one where a house is in a raging fire, and the other is where someone got a slight burn on their finger. Which call should be responded to and helped with first?
<br />
<br />
In emergency response scenarios, time is of the essence. Imagine two incoming messages: one reporting a house engulfed in flames and another describing a minor burn on a finger. Which message should be prioritized? For humans, this decision is intuitive, but for machine learning models, it requires sophisticated algorithms and natural language processing (NLP) techniques to understand context and urgency.
The ability to automatically classify the urgency of support tickets or emergency messages can significantly enhance response times and resource allocation, potentially saving lives and improving service efficiency. Developing such a machine learning (ML) model involves creating an NLP system that can interpret and prioritize textual information based on its urgency.
In this project, we aim to build a robust ML model: the Urgency Classifier. This classifier will analyze text descriptions of support tickets or emergency messages and classify them into different urgency levels. By leveraging the power of transformers and BERT (Bidirectional Encoder Representations from Transformers), our model will be trained to understand the context and urgency embedded within the text.
<br />
<br />
In this project, we decided to build such ML model: the Urgency Classifier.

## 2. Dataset

The dataset used for training and evaluating the urgency classifier model consists of customer support tickets. Each ticket includes a textual description of the issue and a corresponding priority label indicating the urgency of the issue. The dataset is stored in a CSV file with the following columns:

Ticket Description: A textual description of the customer's issue or request.
Ticket Priority: The urgency level of the ticket, which can be one of the following categories: "Low", "Medium", "High", "Critical".

## 3. Structure
* `Files` urgency_classifier.py: The main script for training and evaluating the urgency classifier model. README.md: This readme file.
* `data` data (csv) we used to train model

## 4. Technologies Used and Requirements for Use
* `Python`
* `PyTorch`: open-source machine learning library used for building and training the model.
* `Transformers (Hugging Face)`: A library providing pre-trained transformer models, including BERT.
* `Pandas`: data manipulation and analysis library.
* `scikit-learn`: library used for splitting the dataset into training and testing sets.
* `TQDM`: library for displaying progress bars during training.
