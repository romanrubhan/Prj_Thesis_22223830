from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the T5 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("potsawee/t5-large-generation-squad-QuestionAnswer")
model = AutoModelForSeq2SeqLM.from_pretrained("potsawee/t5-large-generation-squad-QuestionAnswer")

# Provide the input text
input_text = """
Python is a popular programming language known for its simplicity and readability.
It is widely used in web development, data analysis, artificial intelligence, and more.
The syntax of Python is easy to understand, making it a great choice for beginners.
Python supports object-oriented, functional, and imperative programming paradigms.
"""

# Generate questions from the input text
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
outputs = model.generate(**inputs, max_length=100, num_return_sequences=3, num_beams=4, early_stopping=True)

for i, output in enumerate(outputs):
    question_answer = tokenizer.decode(output, skip_special_tokens=False)
    question_answer = question_answer.replace(tokenizer.pad_token, "").replace(tokenizer.eos_token, "")
    question, answer = question_answer.split(tokenizer.sep_token)
    print(f"Question {i + 1}: {question}")


# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#
# # Load the T5 model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained("potsawee/t5-large-generation-squad-QuestionAnswer")
# model = AutoModelForSeq2SeqLM.from_pretrained("potsawee/t5-large-generation-squad-QuestionAnswer")
#
# # Provide the input text
# input_text = """
# Python is a popular programming language known for its simplicity and readability.
# It is widely used in web development, data analysis, artificial intelligence, and more.
# The syntax of Python is easy to understand, making it a great choice for beginners.
# Python supports object-oriented, functional, and imperative programming paradigms.
# """
#
# # Generate questions from the input text
# inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
# outputs = model.generate(**inputs, max_length=100, num_return_sequences=3, num_beams=4, early_stopping=True)
#
# for i, output in enumerate(outputs):
#     question_answer = tokenizer.decode(output, skip_special_tokens=False)
#     question_answer = question_answer.replace(tokenizer.pad_token, "").replace(tokenizer.eos_token, "")
#     question, answer = question_answer.split(tokenizer.sep_token)
#     print(f"Question {i + 1}: {question}")
#
# # from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# #
# # # Load the T5 model and tokenizer
# # tokenizer = AutoTokenizer.from_pretrained("potsawee/t5-large-generation-squad-QuestionAnswer")
# # model = AutoModelForSeq2SeqLM.from_pretrained("potsawee/t5-large-generation-squad-QuestionAnswer")
# #
# # # Provide Python knowledge as context
# # python_knowledge = r"""
# # Supervised learning is the types of machine learning in which machines are trained using well "labelled" training data,
# # and on basis of that data, machines predict the output. The labelled data means some input data is already tagged with the correct output.
# # In supervised learning, the training data provided to the machines work as the supervisor that teaches the machines to predict the output correctly.
# # It applies the same concept as a student learns in the supervision of the teacher.
# # Supervised learning is a process of providing input data as well as correct output data to the machine learning model.
# # The aim of a supervised learning algorithm is to find a mapping function to map the input variable(x) with the output variable(y).
# # In the real-world, supervised learning can be used for Risk Assessment, Image classification, Fraud Detection, spam filtering, etc.
# #
# #
# # #How Supervised Learning Works?
# # In supervised learning, models are trained using labelled dataset, where the model learns about each type of data.
# # Once the training process is completed, the model is tested on the basis of test data (a subset of the training set),
# # and then it predicts the output.
# #
# # #Steps Involved in Supervised Learning:
# # First Determine the type of training dataset
# # Collect/Gather the labelled training data.
# # Split the training dataset into training dataset, test dataset, and validation dataset.
# # Determine the input features of the training dataset, which should have enough knowledge so that the model can accurately predict the output.
# # Determine the suitable algorithm for the model, such as support vector machine, decision tree, etc.
# # Execute the algorithm on the training dataset. Sometimes we need validation sets as the control parameters, which are the subset of training datasets.
# # Evaluate the accuracy of the model by providing the test set. If the model predicts the correct output, which means our model is accurate.
# #
# #
# # # Generate questions from the Python knowledge
# # inputs = tokenizer(python_knowledge, return_tensors="pt")
# # outputs = model.generate(**inputs, max_length=100, num_return_sequences=1, num_beams=5, early_stopping=True)
# #
# # #Types of supervised Machine learning Algorithms:
# # Supervised learning can be further divided into two types of problems:
# #
# # Supervised Machine learning
# # 1. Regression
# #
# # Regression algorithms are used if there is a relationship between the input variable and the output variable.
# # It is used for the prediction of continuous variables, such as Weather forecasting, Market Trends, etc.
# # Below are some popular Regression algorithms which come under supervised learning:
# #
# # Linear Regression
# # Regression Trees
# # Non-Linear Regression
# # Bayesian Linear Regression
# # Polynomial Regression
# #
# #
# # #2. Classification
# # Classification algorithms are used when the output variable is categorical, which means there are two classes such as Yes-No,
# #  Male-Female, True-false, etc.
# # Spam Filtering,
# #
# # Random Forest
# # Decision Trees
# # Logistic Regression
# # Support vector Machines
# #
# #
# #
# # Advantages of Supervised learning:
# # With the help of supervised learning, the model can predict the output on the basis of prior experiences.
# # In supervised learning, we can have an exact idea about the classes of objects.
# # Supervised learning model helps us to solve various real-world problems such as fraud detection, spam filtering, etc.
# #
# #
# # Disadvantages of supervised learning:
# # Supervised learning models are not suitable for handling the complex tasks.
# # Supervised learning cannot predict the correct output if the test data is different from the training dataset.
# # Training required lots of computation times.
# # In supervised learning, we need enough knowledge about the classes of object.
# #
# #
# # """
# #
# # # Split the text into paragraphs or sentences
# # segments = python_knowledge.split("\n\n")  # Split by double newline to separate paragraphs
# #
# # # Generate questions for each segment
# # for segment in segments:
# #     # Generate questions from the segment
# #     inputs = tokenizer(segment, return_tensors="pt", padding=True, truncation=True)
# #     outputs = model.generate(**inputs, max_length=100, num_return_sequences=3, num_beams=4, early_stopping=True)
# #
# #     # Store generated questions without duplicates
# #     generated_questions = set()
# #
# #     for i, output in enumerate(outputs):
# #         question_answer = tokenizer.decode(output, skip_special_tokens=False)
# #         question_answer = question_answer.replace(tokenizer.pad_token, "").replace(tokenizer.eos_token, "")
# #
# #         try:
# #             question, _ = question_answer.split(tokenizer.sep_token, 1)
# #             generated_questions.add(question)
# #         except ValueError:
# #             # Handle the case when separator token is not found
# #             print("Separator token not found in question_answer:", question_answer)
# #
# #     # Print the unique generated questions for this segment
# #     for i, question in enumerate(generated_questions):
# #         print(f"Question {i + 1}: {question}")