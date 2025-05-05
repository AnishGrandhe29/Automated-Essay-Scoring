# Automated Essay Scoring (AES)

Automated Essay Scoring (AES) is a tool developed to evaluate and assign scores to student essays using computer programs. This innovation aims to bring **fairness**, **speed**, and **scalability** to the grading process—allowing educators to focus more on teaching and students to benefit from timely feedback.

## Getting Started

To run the project:

1. Clone this repository.
2. Install the required packages:

   ```bash
   pip install -r requirements.txt
Run the server:

   ```bash
   python manage.py runserver
   ```
Architecture Diagram:

![Image](https://github.com/user-attachments/assets/2f4fb790-6964-4a35-9073-b54562b63c7f)

Dataset
The dataset used is uploaded in the repository as output_data.csv.

Result
The proposed Automated Essay Scoring system was tested using a dataset of labeled essays, specifically focused on automotive topics. After preprocessing, feature extraction, and training, the model achieved promising performance on unseen data.

The system combines deep learning (BiLSTM) with handcrafted features to produce scores that closely match human evaluations.

Key Evaluation Metrics:
Mean Squared Error (MSE): Indicates the average squared difference between predicted and actual scores; lower values imply better performance.

Mean Absolute Error (MAE): Measures average absolute error, offering a more interpretable sense of accuracy.

R² Score: Represents how well the model predictions match the actual values; values closer to 1 indicate higher accuracy.

The model showed consistent performance, with high correlation between predicted and actual scores.

Additional Features:
Accurately identifies whether an essay is relevant to the automotive domain.

Provides detailed feedback on strengths (e.g., strong technical coverage) and areas for improvement (e.g., low environmental topic coverage).

Generates domain-specific suggestions to help users improve their essays effectively.
