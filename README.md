# Mini AI Spam Filter

**Author:** Kyoungjin Oh (2022148070)
**Date:** 2024-12-08

## 1. Introduction

This project addresses the task of SMS Spam Detection, which involves automatically classifying text messages as either legitimate (Ham) or malicious (Spam). This task is particularly interesting and practically important because spam is not merely a nuisance; it often serves as a vector for security threats like phishing. From a university student's perspective, this task is highly relevant because spam is more than just a security risk; it is a constant source of daily disruption. Frequent spam notifications break concentration during study sessions or lectures. More importantly, critical messages—such as updates on team projects, changes in exam schedules, or internship interview offers—can easily get buried under a pile of unwanted advertisements. Developing an AI pipeline for this purpose demonstrates how automated systems can efficiently solve real-world problems that are unscalable for manual moderation.

## 2. Task Definition

### Task Description
The objective of this project is to build a binary text classification system that automatically categorizes SMS messages into two classes:
- **Ham:** Legitimate messages.
- **Spam:** Unwanted or malicious messages.

### Motivation
Spam messages are not only a nuisance that clutters user inboxes but also pose serious security risks, such as phishing and fraud. Manually filtering these messages is time-consuming and inefficient. Therefore, an automated and accurate AI filter is essential to improve user experience and protect users from potential threats.

### Input / Output
- **Input:** A raw text string containing the content of an SMS message (e.g., "URGENT! You have won a 1 week FREE membership...").
- **Output:** A binary label indicating the category: 0 for Ham (Legitimate) or 1 for Spam.

### Success Criteria
The system's performance is evaluated using Precision, Recall, and F1-score. Since the dataset is imbalanced (far more Ham than Spam) and marking a legitimate message as spam (False Positive) is critical, High Precision is a key success factor. Ultimately, the AI pipeline is considered successful if it demonstrates a significantly higher F1-score compared to the rule-based Naïve Baseline.

## 3. Dataset

### Source
I used the 'sms_spam' dataset provided by the Hugging Face datasets library. This is a public set of SMS labeled messages that have been collected for mobile phone spam research.

### Size and Splits
The dataset contains a total of 5,574 examples. I split the dataset into a training set (80%, approx. 4,459 samples) and a test set (20%, approx. 1,115 samples) using a random seed of 42 to ensure reproducibility.

### Preprocessing
Different preprocessing strategies were applied depending on the model architecture.
- **Naïve Baseline:** Input texts were lowercased to ensure case-insensitive matching. Additionally, for the rule generation phase, the training data was tokenized to analyze word frequencies and extract discriminative keywords.
- **AI Pipeline:** Raw text was processed using the Sentence Transformer (all-MiniLM-L6-v2) model, which handles tokenization, normalization, and converts sentences into dense embedding vectors.

## 4. Methods

### 4.1 Naïve Baseline

**Method Description**
Instead of manually guessing keywords, I implemented a data-driven heuristic method using frequency analysis tools from Scikit-Learn. I analyzed the training corpus to calculate the frequency difference between spam and ham messages for every word (Score = Frequency in Spam - Frequency in Ham). The top 10 words with the highest scores—indicating they appear frequently in spam but rarely in legitimate messages—were selected as the "spam dictionary" (e.g., free, txt, claim). The baseline model classifies a message as spam solely if it contains any of these extracted keywords.

**Why Naïve**
This approach is considered naïve because it ignores context, sentence structure, and other features like punctuation or capitalization. It relies exclusively on a simple "bag-of-words" assumption, treating the presence of a specific word as the sole determinant for classification regardless of how it is used in the sentence. It cannot handle polysemy (words with multiple meanings) or negation.

**Likely Failure Modes**
- **Contextual False Positives:** Legitimate messages containing trigger words used in a benign context will be misclassified. For example, "Are you free for dinner?" would be incorrectly marked as spam because "free" is a top keyword.
- **Keyword Evasion (False Negatives):** Spam messages that do not use the specific top 10 keywords (e.g., using synonyms or creative misspellings like "F-R-E-E") will be completely missed, as there are no backup rules to catch them.

### 4.2 AI Pipeline

**Models Used**
- **Embedding:** sentence-transformers/all-MiniLM-L6-v2
- **Classifier:** Logistic Regression

**Pipeline Stages**
1. **Preprocessing:** The raw input text is handled by the SentenceTransformer tokenizer. This step automatically performs lowercasing, tokenization (splitting text into sub-words), and adds special tokens required by the BERT architecture.
2. **Representation (Embedding):** The pre-trained all-MiniLM-L6-v2 model processes the tokenized input. It converts the variable-length text into a fixed-size 384-dimensional dense vector. This vector captures the semantic meaning of the sentence.
3. **Decision Component:** The extracted 384-dimensional vectors are fed into a Logistic Regression classifier. The classifier calculates the probability of the vector belonging to the "Spam" class based on the decision boundary learned during training.
4. **Post-processing:** The system applies a standard threshold (default 0.5) to the probability score. If the probability is greater than 0.5, the label 1 (Spam) is returned; otherwise, 0 (Ham) is returned.

**Design Choices and Justification**
- **Efficiency vs. Performance:** I selected all-MiniLM-L6-v2 because it offers an optimal balance between speed and accuracy. It is significantly faster and smaller (approx. 80MB) than full-sized BERT models, making it suitable for a lightweight pipeline while maintaining high semantic understanding capabilities.
- **Transfer Learning Approach:** Instead of training a model from scratch or fine-tuning a large transformer (which requires significant compute), I used the pre-trained model as a feature extractor. Since the pre-trained model already "knows" English semantics well, combining it with a simple linear classifier (Logistic Regression) is sufficient to achieve high performance on this small dataset without overfitting.

## 5. Experiments and Results

### Metrics
To evaluate the performance of the spam classification models, I selected Precision, Recall, and F1-score. Since the dataset is highly imbalanced—containing significantly more legitimate ("Ham") messages than "Spam"—standard Accuracy can be misleading. Therefore, the selected metrics align with the specific goals of this task as follows:
- **Precision:** This measures the proportion of correctly identified spam messages among all messages predicted as spam. For a spam filter, Precision is the most critical metric aligned with user trust. A False Positive (classifying a legitimate email as spam) is a critical failure. High precision ensures the system is safe to use.
- **Recall:** This measures the proportion of actual spam messages that the model successfully detected. While important for keeping the inbox clean, a False Negative is generally considered a minor annoyance rather than a critical failure compared to a False Positive.
- **F1-Score:** The F1-score is the harmonic mean of Precision and Recall. It provides a single, balanced metric to fairly compare the Naïve Baseline and the AI Pipeline.

### Results
The performance of the Naïve Baseline and the AI Pipeline was evaluated on the held-out test set. The results demonstrate that the AI Pipeline significantly outperforms the baseline across all key metrics.

**Table 1: Performance comparison results**

| Method | Precision | Recall | F1-score |
| :--- | :--- | :--- | :--- |
| Naïve Baseline | 0.79 | 0.81 | 0.80 |
| AI Pipeline | 1.00 | 0.91 | 0.95 |

The AI Pipeline achieved a **perfect Precision of 1.00** and a significantly higher **F1-score of 0.95**. This indicates that the semantic understanding provided by the MiniLM embeddings allowed the model to effectively distinguish between spam and legitimate messages, even when they shared common vocabulary. The bar chart (performance_comparison.png) clearly highlights the gap in F1-score, and the confusion matrices (confusion_matrices.png) reveal that the AI Pipeline drastically reduced the number of False Positives compared to the baseline.

### Qualitative Examples (Case Study)
To better understand the model behaviors, I analyzed specific cases where the two methods produced different predictions.

- **Example 1: Contextual Understanding (False Positive in Baseline)**
  - **Text:** "K, wen ur free come to my home and also tel vikky i hav sent mail to him also.. Better come evening il be free today aftr 6pm..:-)"
  - **Actual:** Ham
  - **Baseline Prediction:** Spam (Incorrect)
  - **AI Pipeline Prediction:** Ham (Correct)
  - **Analysis:** The Baseline model incorrectly flagged this message as spam, likely triggered by the keyword "free." The AI Pipeline, however, understood the conversational context (making plans to meet at home) and correctly identified it as legitimate.

- **Example 2: Semantic Detection (False Negative in Baseline)**
  - **Text:** "Call Germany for only 1 pence per minute! Call from a fixed line via access number 0844 861 85 85. No prepayment. Direct access!"
  - **Actual:** Spam
  - **Baseline Prediction:** Ham (Incorrect)
  - **AI Pipeline Prediction:** Spam (Correct)
  - **Analysis:** The Baseline failed to detect this spam because the specific words (e.g., "Germany", "pence", "0844") were likely not in the top-10 keyword list. The AI Pipeline successfully captured the semantic meaning of a promotional offer ("Call... for only...") through its embedding representation.

- **Example 3: Handling Ambiguity (False Positive in Baseline)**
  - **Text:** "Dear,shall mail tonite.busy in the street,shall update you tonite.things are looking ok.varunnathu edukkukayee raksha ollu.but a good one in real sense."
  - **Actual:** Ham
  - **Baseline Prediction:** Spam (Incorrect)
  - **AI Pipeline Prediction:** Ham (Correct)
  - **Analysis:** The Baseline was likely triggered by the word "mail," which often appears in spam contexts. The AI Pipeline correctly handled the informal grammar, mixed language, and typos, recognizing the message as a personal update rather than a commercial blast.

## 6. Reflection and Limitations

**Reflection**
The AI pipeline worked better than expected, achieving a perfect Precision of 1.00 on the test set, which demonstrates that even a lightweight model like all-MiniLM-L6-v2 can effectively capture semantic nuances that simple keywords miss. However, the implementation process presented unexpected technical challenges. Although I attempted to load the model following the standard instructions provided by Hugging Face, I encountered persistent errors likely due to version conflicts or environment incompatibilities. Troubleshooting these integration issues required significant time and effort. In terms of evaluation, standard accuracy proved to be a misleading metric due to the significant class imbalance (mostly Ham); therefore, Precision and F1-score were far more effective in capturing the "quality" of the system by penalizing False Positives.

**Limitations**
The primary limitation of the Naïve Baseline was its inability to understand context, leading to failures where benign words like "free" triggered false alarms. Regarding the AI pipeline, although it uses a "frozen" pre-trained encoder with a linear classifier, it achieved a perfect Precision of 1.00. However, there is still room for improvement in Recall (0.91), indicating that the model occasionally misses subtle spam messages. Additionally, the model might be vulnerable to adversarial attacks (e.g., intentional misspellings like "F-R-E-E"). If I had more time and compute resources, I would try fine-tuning the entire Transformer model end-to-end to capture these missed cases and boost Recall, while also exploring data augmentation techniques to improve robustness against such evasion tactics.
