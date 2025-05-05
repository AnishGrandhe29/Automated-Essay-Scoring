# Automated-Essay-Scoring
Automated Essay Scoring (AES) is a tool developed to evaluate and assign scores  to student essays using computer programs. This innovation aims to bring fairness,  speed, and scalability to the grading process, allowing educators to focus more on  teaching and students to benefit from timely feedback.


To run the project, clone this repository then download all the requirements from requirements.txt, and run using the command python manage.py runserver

Architecture Diagram:

![Image](https://github.com/user-attachments/assets/2f4fb790-6964-4a35-9073-b54562b63c7f)

The dataset used is uploaded in the github repository which is output_data.csv.

Result: 
The proposed Automated Essay Scoring system was tested using a dataset of 
labeled essays, specifically focused on automotive topics. After preprocessing, 
feature extraction, and training, the model achieved promising performance on 
unseen data. The system combined deep learning (BiLSTM) with handcrafted 
features to produce scores that closely matched human evaluations. 
Key evaluation metrics: 
● Mean Squared Error (MSE): Indicates the average squared difference 
between predicted and actual scores; lower values imply better 
performance. 
● Mean Absolute Error (MAE): Measures average absolute error, offering a 
more interpretable sense of accuracy. 
● R² Score: Represents how well the model predictions match the actual 
values; values closer to 1 indicate higher accuracy. 
The model showed consistent performance, with high correlation between 
predicted and actual scores. Additionally, the system was able to: 
● Accurately identify whether an essay was relevant to the automotive 
domain. 
● Provide detailed feedback on strengths (e.g., strong technical coverage) and 
improvement areas (e.g., low environmental topic coverage). 
● Generate domain-specific suggestions that help users improve their essays 
effectively. 
These results demonstrate that the system is reliable, scalable, and capable of 
providing both scores and qualitative feedback in real-time. 
