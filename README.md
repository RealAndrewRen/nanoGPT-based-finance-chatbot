# Pennywise - A Financial Literacy Large Language Model

### 👥 **Team Members**

| Name             | GitHub Handle | Contribution                                                             |
|------------------|---------------|--------------------------------------------------------------------------|
| Andrew Ren    | @realandrewren | Data Preprocessing, Model Training, Project Coordination                   |
| Nailya Alimova   | @naiilya     | Exploratory Data Analysis, Supervised Fine-Tuning, Project Management     |
| Anjali Amin     | @anjali5582  | Data Preprocessing, Data Validation, Results Interpretation                |
| Marvin Hoang      | @marhvin       | Model Selection, Hyperparameter Tuning, Model Training & Optimization  |
| Naisha Mistry       | @naishahmistry    | Model Training & Evaluation, Performance Analysis, Results Interpretation |   

---

## 🎯 **Project Highlights**

- Developed a machine learning model using a GPT-style transformer fine-tuned with supervised instruction data to address the challenge of building a finance-domain conversational assistant.

- Achieved substantial improvements in domain-specific response quality compared to the base model, demonstrating strong applicability for finance-related support.

- Generated actionable insights by analyzing conversation patterns and user intents, helping inform feature planning and product direction.

- Implemented a structured SFT (Supervised Fine-Tuning) pipeline optimized for small-to-medium datasets, addressing constraints around limited compute and reproducibility expectations.

---

## 👩🏽‍💻 Setup and Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/finance-nanogpt.git
cd finance-nanogpt
```

### 2️⃣ Install Dependencies

```bash
pip install tiktoken transformers datasets tqdm torch
```

For GPU acceleration, ensure you have a CUDA-compatible PyTorch installation.

### 3️⃣ Environment Setup (Optional but Recommended)

Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate     # macOS/Linux
venv\Scripts\activate        # Windows
```
### 4️⃣ Load Datasets (Optional)

The training pipeline currently supports Hugging Face datasets only.

To add a custom dataset, insert an entry with the following format:
```json
{"name": "<dataset_name>", "cols_to_parse": ["<columns>"], "split": "train"}
```
- For domain-adaptation pretraining, edit:
```bash
  data/finance_data/prepare.py
```
- For supervised fine-tuning (SFT), edit:
```bash
  data/finance_data/sft_prepare.py
```

When adding SFT datasets, ensure the `format_sft` function tokenizes the correct columns.

### 5️⃣ Encode Training Binaries (Domain Adaptation Pretraining)
```bash
python data/finance_data/prepare.py
```
### 6️⃣ Domain Adaptation Pretraining
```bash
python train.py config/train_financedata.py
```
Training typically runs for 5k–20k iterations, depending on available compute.
The resulting model checkpoint will be saved in:
```bash
out-finance_data/ck.pt
```

### 7️⃣ Encode Binaries for Supervised Fine-Tuning
```bash
python data/finance_data/sft_prepare.py
```
### 8️⃣ Supervised Fine-Tuning (SFT)

1. In train.py, update the initialization method:
   ```python
   init_from = "resume"
   ```
   (replace the default gpt2 initialization)

2. Execute Steps 1–5 of:
   ```bash
   data/finance_data/sft_training_colab.ipynb
   ```
3. Train for at least 4k iterations, then save the resulting `ck.pt` checkpoint.

   # Note: Colab is currently required for this step. Work is underway to fully separate this notebook into executable scripts.

### 9️⃣ Prompt the Model

Follow the remaining steps in:
```bash
data/finance_data/sft_training_colab.ipynb
```
to load the trained checkpoint and interactively prompt the fine-tuned model.

---

## 📊 **Project Overview**

- This project was developed by a team of five AI Studio Fellows at MIT as part of the Break Through Tech AI Program, under the supervision of David Fang (Member of Technical Staff, OpenAI).
- Our team focuses on building a personalized GPT-2 Small (124M) language model using the nanoGPT framework.
- The primary objective is to develop a financial-literacy chatbot designed for young users, particularly students, who are seeking accessible explanations of finance concepts such as budgeting, investing, and core economic terminology.

To achieve this, we implemented a two-stage training pipeline:
1. Domain-adaptive pretraining on large-scale financial text corpora
2. Supervised fine-tuning (SFT) to enable conversational, instruction-following behavior

This approach allows the model to first acquire strong domain knowledge in finance and then adapt that knowledge into coherent, user-friendly responses suitable for real-world Q&A interactions.

---

## 📊 **Data Exploration**

1. Base (Pretraining) Datasets
   
For initial domain adaptation, we curated a diverse set of finance-focused text datasets:
- Financial textbooks: https://huggingface.co/datasets/alvanlii/finance-textbooks/viewer/default/train?row=0&views%5B%5D=train 
- Bloomberg financial news: https://huggingface.co/datasets/genloop/bloomberg_financial_news_120k 
- Financial news articles: https://huggingface.co/datasets/ashraq/financial-news-articles 
- Aggregated financial news corpora: https://huggingface.co/datasets/edaschau/financial_news/discussions

These datasets were selected to ensure coverage across formal academic writing, professional news reporting, and general explanatory finance content. 
This balance enables the model to respond effectively to both technical finance questions and everyday financial inquiries.

Preprocessing Pipeline:
- Used regex-based cleaning to remove URLs, timestamps, and redundant source metadata
- Dropped non-essential columns, retaining only core financial text
- Converted cleaned text into .txt files compatible with nanoGPT
- Tokenized text to transform words into model-readable tokens

2. Supervised Fine-Tuning (SFT) Datasets

After obtaining a pretrained financial checkpoint, we performed supervised fine-tuning using:
- Finance Instruct: https://huggingface.co/datasets/Josephgflowers/Finance-Instruct-500k 
- Finance Alpaca: https://huggingface.co/datasets/gbharti/finance-alpaca/viewer/default/train?row=0&views%5B%5D=train 

These datasets were chosen for their high-quality question-answer and instruction-response pairs, spanning academic finance and personal finance topics. Together, they support robust conversational performance across both formal and informal user queries.

Preprocessing Pipeline:
- Applied regex cleaning to remove rows containing LaTeX artifacts and profanity
- Inserted explicit “User” and “Assistant” tokens to structure dialogue
- Masked “User” tokens so loss is computed only on “Assistant” outputs
- Converted cleaned conversations into .txt format compatible with nanoGPT
- Tokenized text to transform words into model-readable tokens

---

## 🧠 **Model Development**

### Base Model Architecture
- Base model: GPT-2 Small (123.65M parameters)
- Rationale: lightweight, fast to train, well-documented for domain adaptation, and ideal for small applications or teaching

### Pre-Training for Domain Adaptation
**Goal: Teach the base GPT-2 model to understand fundamental financial concepts and build general domain knowledge around finance/economics**
- Initialized from the GPT-2 Small (123.65M) pretrained checkpoint
- Trained on the processed financial pre-training datasets mentioned above
- Hyperparameters Tuned
  - Number of training iterations: 6k, 20k, 50k
- Evaluation Metrics
  - Coherence, Accuracy, and Relevance
  - All metrics were measured by manually evaluating responses to the following prompt: “Why can’t we print more money? Because inflation…”

### Supervised Fine-Tuning (SFT)
**Goal: Enable chatbot-style functionality by training the model to provide structured, straightforward, and easy-to-understand answers to personal finance questions**
- Fine-tuned by initializing from the 20k-iteration checkpoint from pre-training
- Trained on the processed SFT datasets mentioned agove
- Hyperparameters Tuned:
  - Number of training iterations: 4.3k, 10k
- Evaluation Metrics
  - Coherence, Accuracy, and Relevance
  - All metrics were measured by manually evaluating responses to the following prompt:  “USER: What are stocks and how should i start trading them?”

---

## 📈 **Results & Key Findings**

### Summary of Key Findings
- nanoGPT-20k is the best-performing pretrained model, as it delivers the most coherent and relevant responses compared to nanoGPT-6k and nanoGPT-50k.
- nanoGPT-6k outperforms nanoGPT-50k, as it suffers from overfitting and produces degraded/hallucinated outputs.
- SFT fine-tuning enhances nanoGPT-20k by generating clear, conversational, chatbot-style responses that remain focused on the user’s question
- While SFT results show improvement in accuracy and relevance, the generated outputs can be improved by addressing all aspects of the user's question in detail and avoiding circular reasoning patterns


### Pre-Training Results

#### Pretrained Model Comparison Summary

| Name            | Description     | Results                                                         | Pros | Cons |
|-----------------|-------------------|------------------------------------------------------------------|------|------|
| **nanoGPT-6k**      | Custom GPT trained on finance datasets for 6,000 iterations (123.65M parameters) | Professional economic analysis with somewhat coherent discussion of Fed policy, bond markets, and inflation dynamics | • Maintains logical flow and topic relevance<br>• References real economic indicators<br>• Professional analytical tone<br>• Minimal hallucinations | • Limited training may miss nuanced patterns<br>• May lack conversational variety<br>• Shorter responses than expected<br>• Occasional grammatical issues |
| **nanoGPT-20k** | Same model architecture trained for 20,000 iterations | Professional, article-style response with coherent, reputable, and on-topic discussions on Fed policies, bonds, and inflation dynamics | • Professional & engaging tone<br>• Uses appropriate financial vocabulary<br>• Mentions accurate metrics/statistics<br>• Draws accurate economic connections | • Occasional grammar/logic drift <br>• Risk of hallucinations |
| **nanoGPT-50k**    | Same model architecture trained for 50,000 iterations showing overtraining effects | Degraded responses with topic drift, personal anecdotes, and hallucinations | • Captures conversational finance-forum style<br>• Uses appropriate financial vocabulary<br>• Longer, more detailed responses<br>• More grammatically correct | • Severe overtraining symptoms<br>• Frequent topic drift<br>• Hallucinated content / fake data<br>• Circular reasoning patterns<br>• Poor factual accuracy |

---
**nanoGPT-6k Results**

**Sample Output**
> **Prompt:** *“Why can’t we print more money? Because inflation…”*  
> **Model Output:**  
> *“Investors are that means lower-term uncertainty and a little bit of uncertainty in the political landscape is going to likely cause the Federal Reserve to raise rates four steps this year. The Fed has said it is also raising rates three times this year. That has helped buoy the bond market as bond yields have risen. But that has brought uncertainty to the market, and investors are concerned about an increase in the pace of the inflation rate.”*

**Performance Evaluation**

| Metric      | Rating | Notes |
|-------------|--------|-------|
| **Coherence** | 3/5 | Maintains logical flow, Professional economic analysis style, Generates grammatically incorrect phrases |
| **Accuracy**  | 4/5 | References real economic data (Fed rates, bond yields) |
| **Relevance** | 4/5 | Stays aligned with inflation and monetary policy with minimal topic drift |

--- 
**nanoGPT-20k Results**
**Sample Output**
> **Prompt:** *“Why can’t we print more money? Because inflation…”*  
> **Model Output:**  
> *“rise below 2 percent, and that's when a big boom will begin. The question is, does that happen? Bill Gross , founder of the global asset manager where all of his money has gone, says it's hard to get ahead of inflation. "If you get to that level, you're going to see that we'll see a big spike in the interest rates and that's probably going to be a real catalyst for the Fed," he says. The Fed is set to release minutes from its June meeting this week and could make a decision on whether to increase or reduce its monthly bond purchases at any time in the future. The Fed is expected to hold its policy interest rates steady and it could take another six to 12 months before it lifts its benchmark rate, the Fed's most-predictable outlook. But as for whether and how much inflation has to rise before it could push down the Fed, according to Gross, it doesn't appear anyone knows for sure. "Any time you have the big price declines, the Fed will take action and it's going to take a long time, because inflation is going to rise," Gross says.”*

**Performance Evaluation**

| Metric    | Rating |  Notes |
|-------------|--------|-------|
| **Coherence** | 4/5 | Provides a smooth segue from prompt to first word of generated response, Style and formality of a news article, Maintains logical flow with little to no drift in grammar|
| **Accuracy**  | 3.5/5 | Accurate inflation threshold, References relevant economic concepts (Federal interest rates, bond purchases), Shows understanding of how inflation affects interest rate policies and Federal Reserve’s bond-buying actions, Potentially hallucinating quotes and names |
| **Relevance** | 4/5 | Maintains clear focus on inflation, monetary policy, and Fed decisions with minimal topic drift |

---
**nanoGPT-50k Results**

**Sample Output**
> **Prompt:** *“Why can’t we print more money? Because inflation…”*  
> **Model Output:**  
> *“Assistant: Does the value of your mortgage actually hit the value of your property?
Human: I'm not sure, it's only $500 a month. Say the value of your home is $100k. That's what I assume. You actually make $85k. That's probably better than what you paid for a single property and you can afford a condo with some reasonable interest rates. So you can afford something with a 10% down payment on a home that comes with a $500k down payment.”*

**Performance Evaluation**

| Metric      | Rating | Notes |
|-------------|--------|-------|
| **Coherence** | 2/5 | Circular reasoning patterns, Fragmented dialogue, Forum-style response with degraded quality |
| **Accuracy**  | 1/5 | Mixes correct/incorrect information, generates hallucinations (overtrained)|
| **Relevance** | 1/5 | Immediate topic drift from inflation and monetary policy |


### SFT Results

> **Prompt:** *“USER: What are stocks and how should I start trading them?”*  
> **Model Output:**  
> *“ASSISTANT: Stock trading can be done through a variety of platforms, from initial share sales to trades on your own platform. It is important to consider your own investment strategy and keep a close eye on your own trading behavior. Additionally, you should consider creating a diversified portfolio of stocks that can benefit from your unique skillset.
Stock trading can be challenging and involves many different types of investments and can be complex, time consuming and challenging. Here are some tips to help you manage your investments: 1. Create a diversified portfolio of stocks: Once you have a diversified portfolio of stocks, it is important to consider investments from various sectors to identify opportunities. It is important to prioritize diversification and diversification over inflation and high unemployment levels. 2. Set a clear timeline: Start trading your own stocks at the beginning of the year, and track your progress in developing your own stocks. It is important to prioritize the main goal of your investment and make sure you are comfortable with the direction of your investment.”*

**Performance Evaluation**

| Metric      | Rating | Notes |
|-------------|--------|-------|
| **Coherence** | 3/5 | Has a conversational and logical flow of ideas that answers the prompt, Occasionally drifts in grammar or provides circular reasoning, Does not fully address the second half of the user's prompt|
| **Accuracy**  | 4/5 | Generates realistic and accurate tips related to the prompt based on the second part of the user’s question |
| **Relevance** | 5/5 | Information does not stray from the prompt’s topic |

---

## 🚀 **Next Steps**

With additional time and resources, we would pursue the following directions to further improve our model’s performance, usability, and real-world impact:

### Expanded Data Collection via Web Scraping
- Scrape high-quality, reputable personal-finance educational resources (e.g., Khan Academy, Fidelity, FDIC, Investopedia) to broaden the model’s exposure to practical, user-facing financial explanations.
- Focus on beginner-friendly content covering budgeting, credit, investing fundamentals, and financial safety to better align with the target audience.
- Apply automated filtering, deduplication, and quality checks to ensure clean, domain-relevant training data.

### Large-Scale Training and Fine-Tuning
- Refine prompting and loss-masking strategies to better isolate assistant responses and reduce circular reasoning.
- Scale training to longer context lengths and experiment with larger foundation models beyond GPT-2 Small.
- Leverage cloud-based infrastructure and higher-performance GPUs to support longer training runs and improved convergence.
- Explore training a model from scratch rather than initializing from GPT-2 to reduce inherited biases and better tailor the architecture to finance-domain objectives.

### Reinforcement Learning from Human Feedback (RLHF)
- Apply reinforcement learning from human feedback to improve alignment, factual accuracy, and instructional clarity.
- Collect structured human feedback on model responses using the new [**nanoChat**](https://github.com/karpathy/nanochat) repository.
- Fine-tune the model to prefer clear, concise, and actionable financial guidance while minimizing hallucinations.

### User-Facing Application Development
- Build an interactive, user-facing interface that allows users to chat with Pennywise in real time.
- Incorporate features such as conversation history, topic-based prompts, and age-appropriate explanations.
- Use anonymized user interaction data to inform future evaluation and iterative training cycles.

---

## 📝 **License**

If applicable, indicate how your project can be used by others by specifying and linking to an open source license type (e.g., MIT, Apache 2.0). Make sure your Challenge Advisor approves of the selected license type.

**Example:**
This project is licensed under the MIT License.

---

## 📄 **References** (Optional but encouraged)

Cite relevant papers, articles, or resources that supported your project.

---

## 🙏 **Acknowledgements** 

We would like to thank our Challenge Advisor David Fang who is a member of technical staff at OpenAI and Rawisara (Mimi) Lohanimit who was our TA! Their guidance, encouragement, and technical insight were invaluable throughout the development of this project. We are especially thankful for the time and effort they volunteered in support of the Break Through Tech program. They played a significant role in the development of this project and we are very grateful for the effort they put into our learning as students. 
