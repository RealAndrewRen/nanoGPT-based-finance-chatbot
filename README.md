# Pennywise - A Financial Literacy Large Language Model

### 👥 **Team Members**

**Example:**

| Name             | GitHub Handle | Contribution                                                             |
|------------------|---------------|--------------------------------------------------------------------------|
| Andrew Ren    | @realandrewren | Data preprocessing, model training, overall project coordination            |
| Nailya Alimova   | @naiilya     | Data collection, exploratory data analysis (EDA), dataset documentation  |
| Anjali Amin     | @anjali5582  | Data preprocessing, feature engineering, data validation                 |
| Marvin Hoang      | @marhvin       | Model selection, hyperparameter tuning, model training and optimization  |
| Naisha Mistry       | @naishahmistry    | Model evaluation, performance analysis, results interpretation           |

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

python data/finance_data/prepare.py

### 6️⃣ Domain Adaptation Pretraining
```text
python train.py config/train_financedata.py
```
Training typically runs for 5k–20k iterations, depending on available compute.
The resulting model checkpoint will be saved as ck.pt.

### 7️⃣ Encode Binaries for Supervised Fine-Tuning
```text
python data/finance_data/sft_prepare.py
```
### 8️⃣ Supervised Fine-Tuning (SFT)

1. In train.py, update the initialization method:
   ```python
   init_from = "resume"
   ```
   (replace the default gpt2 initialization)

3. Resume training from the domain-adapted checkpoint and train for at least 4k iterations.

4. Save the resulting checkpoint (ck.pt) upon completion.

### 9️⃣ Prompt the Model

Use the generation or inference scripts to interact with the trained model.

Work is underway to fully replace remaining notebook-based steps with standalone scripts for improved reproducibility.



---

## 🏗️ **Project Overview**

**Describe:**

- How this project is connected to the Break Through Tech AI Program
- Your AI Studio host company and the project objective and scope
- The real-world significance of the problem and the potential impact of your work

---

## 📊 **Data Exploration**

**You might consider describing the following (as applicable):**

* The dataset(s) used: origin, format, size, type of data
* Data exploration and preprocessing approaches
* Insights from your Exploratory Data Analysis (EDA)
* Challenges and assumptions when working with the dataset(s)

**Potential visualizations to include:**

* Plots, charts, heatmaps, feature visualizations, sample dataset images

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

**You might consider addressing the following (as applicable):**

* What are some of the limitations of your model?
* What would you do differently with more time/resources?
* What additional datasets or techniques would you explore?

---

## 📝 **License**

If applicable, indicate how your project can be used by others by specifying and linking to an open source license type (e.g., MIT, Apache 2.0). Make sure your Challenge Advisor approves of the selected license type.

**Example:**
This project is licensed under the MIT License.

---

## 📄 **References** (Optional but encouraged)

Cite relevant papers, articles, or resources that supported your project.

---

## 🙏 **Acknowledgements** (Optional but encouraged)

Thank your Challenge Advisor, host company representatives, TA, and others who supported your project.
