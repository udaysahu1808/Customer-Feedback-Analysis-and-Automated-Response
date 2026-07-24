# Customer Feedback Analysis and Automated Response

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Data Visualization](https://img.shields.io/badge/Data%20Visualization-Matplotlib-11557C)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0)
![WordCloud](https://img.shields.io/badge/WordCloud-Text%20Visualization-8E44AD)
![Rule-Based NLP](https://img.shields.io/badge/Approach-Rule--Based%20%28No%20ML%29-orange)
![Generative AI](https://img.shields.io/badge/Generative%20AI-Grok%20API-4B0082?style=for-the-badge&logo=x&logoColor=white)
![EDA](https://img.shields.io/badge/EDA-Exploratory%20Data%20Analysis-6A1B9A)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

A transparent, rule based Python pipeline that transforms 10,000 unread restaurant reviews into a prioritized action list automatically flagging **critical (1–2 star) reviews**, extracting recurring **complaint themes** and using a **Generative AI model** to draft ready to send apology emails for the most urgent cases.

---

## 📌 Table of Contents

- [Project Introduction](#project-introduction)
- [Problem Statement](#-problem-statement)
- [Objectives of the Study](#-objectives-of-the-study)
- [Scope of the Study](#-scope-of-the-study)
- [Significance of the Study](#-significance-of-the-study)
- [Dataset](#-dataset)
- [Project Structure](#project-structure)
- [Analysis Phases](#analysis-phases)
- [Key Insights](#key-insights)
- [Strategic Recommendations](#strategic-recommendations)
- [Executive Summary, Future Scope & Conclusion](#phase-911-executive-summary-future-scope--conclusion)
- [Tools & Technologies](#️-tools--technologies)
- [How to Run](#️-how-to-run)
- [Summary Table](#summary-table)
- [License](#-license)
- [Author](#-author)

---

## Project Introduction

Retail and hospitality businesses collect large volumes of customer reviews every week, but most of that feedback sits unread until someone manually sifts through it looking for the reviews that need urgent attention. This project builds a small, transparent Python pipeline that does that sifting automatically flagging the most critical (low-rated) reviews, surfacing the recurring complaint themes behind them and using a Generative AI model to draft ready to send apology emails for the worst cases.

By combining rule-based filtering, keyword-frequency analysis and AI-assisted response generation into a single workflow, the project shows that a support team can go from "10,000 unread reviews" to "here are the ones that matter, here's why and here's a draft response" in a few seconds without training a single machine learning model.

---

## 📖 Problem Statement

A company receives thousands of customer reviews and support tickets every week. Currently, the support team manually reads through all of them to find the urgent, negative reviews and drafts individual apology emails by hand. This is:

- **Slow** : Doesn't scale as review volume grows.
- **Inconsistent** : Response quality depends on who's writing that day.
- **Risky** : The backlog of unaddressed critical feedback grows, and genuinely urgent complaints risk getting lost in the noise or answered too late to matter.

Therefore, there is a need for a lightweight, explainable system that can automatically isolate the most urgent feedback, explain *why* it's urgent and accelerate the response process, without requiring a full ML/NLP pipeline.

---

## 🎯 Objectives of the Study

**Primary Objective:**
To develop a Python-based, rule-first customer feedback triage system that identifies critical reviews, explains the drivers behind them and generates AI-assisted response drafts.

**Specific Objectives:**
- To load and clean a raw customer review dataset (text reviews + star ratings).
- To use simple, transparent rule-based logic **no machine learning** to isolate "critical" (1–2 star) reviews.
- To identify the most common complaint keywords within critical reviews using frequency analysis.
- To group complaint keywords into named, business-relevant categories (Service, Food Quality, Pricing, Delivery, Ambience, Hygiene).
- To use a Generative AI API to automatically draft short, personalized, empathetic apology emails for the most critical, most detailed negative reviews.
- To document the approach, limitations and run instructions for reuse and extension.

---

## 📚 Scope of the Study

**In scope:**
- Loading, cleaning, and preparing a single review dataset (text + rating) with pandas.
- Rule-based filtering of critical reviews and keyword frequency analysis using basic Python (`collections.Counter`, string methods) explicitly no ML/NLP models.
- Selecting a small sample (3) of critical, detailed reviews and generating apology emails via a Generative AI API (Grok, with OpenAI/Gemini as alternatives).
- Documenting the approach, limitations and run instructions.

**Out of scope:**
- Any machine learning or NLP modeling (e.g., sentiment classifiers, topic modeling) the brief explicitly calls for rule-based logic instead.
- Automatically sending emails (the notebook only drafts them; sending would require integrating an email/SMTP or ticketing system).
- Handling non-English reviews, multi-language support, or negation-aware/semantic keyword extraction (noted as a limitation, not solved here).
- Production concerns like scaling to real-time review streams, deduplication across platforms or a persistent database — this is a single-run analysis notebook, not a deployed service.

---

## 📚 Significance of the Study

This project shows that critical customer feedback can be triaged and responded to faster and more consistently, without needing machine learning or heavy infrastructure. It is beneficial for:

- **Support teams** : Reduces manual workload and standardizes response quality.
- **Management** : Provides a quick, quantified read on recurring problem areas (e.g., service, food quality).
- **Customers** : Faster, more consistent acknowledgment improves retention of unhappy customers.
- **Analysts & students** : A simple, auditable template for feedback triage that's easy to extend to other review or ticket sources.

---

## 📊 Dataset

**File:** `Restaurant reviews.csv`

| Column | Description |
|---|---|
| `Reviewer` | Name of the customer who left the review |
| `Review` | Raw review text |
| `Rating` | Star rating given (1–5, with some inconsistent entries like `"Like"` and decimals) |
| `Metadata`, `Pictures`, `7514` | Dropped during cleaning (not used in the analysis) |

**Size:** ~10,000 restaurant reviews (post-cleaning)

> **Note:** The notebook currently loads data from a local Windows path (`C:\Users\udays\Downloads\Restaurant reviews.csv`). Update this path to point to your local `Dataset/` folder before running.

---

## Project Structure

```
Restaurant-Customer-Feedback-Analysis/
│
├── Dataset/
│   └── Restaurant reviews.csv
│
├── Notebook/
│   └── Customer Feedback Analysis.ipynb
│
├── Deliverables/
│   ├── Automated AI Generated Emails.txt
│   ├── Complaint Category Summary.xlsx
│   └── Critical Restaurant Reviews.xlsx
│
├── Visuals/
│   ├── Customer Rating Distribution.png
│   ├── Percentage of Critical Reviews.png
│   ├── Top Complaints Keywords.png
│   ├── Word Cloud.png
│   └── Placeholder.txt
│
├── README.md
├── LICENSE
└── requirements.txt
```

---

## Analysis Phases

### Phase 1: Data Collection & Understanding
Import libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `re`, `os`, `Counter`, `wordcloud`); load the CSV; inspect shape, columns, dtypes, and summary statistics; drop the invalid `"Like"` rating category and cast `Rating` to float.

### Phase 2: Missing Value Analysis
Identify missing values per column and visualize them with an annotated bar chart.

### Phase 3: Dropping & Duplicate Investigation
Drop irrelevant columns (`7514`, `Metadata`, `Pictures`), drop null rows, detect and remove duplicate rows, and apply a text-cleaning function (lowercase, strip non-alphabetic characters, collapse whitespace) to create a `Clean Review` column.

### Phase 4: Exploratory Data Analysis (EDA)
Visualize the distribution of star ratings across the dataset.

**Insight:** 5-star reviews (3,826) are most frequent, followed by 4-star (2,373); 1-star reviews (1,735) still represent a notable share of dissatisfaction warranting investigation. The overall distribution is skewed toward higher ratings.

### Phase 5: Rule-Based Filtering
Define "critical" reviews as those with `Rating <= 2` (no ML — a single conditional filter); compute the count and percentage of critical reviews relative to the full dataset.

### Phase 6: Complaint Keyword Extraction
Remove common stop words; count word frequency across critical reviews with `Counter`; extract the top 20 complaint keywords; group keywords into **6 named complaint categories** and tally how many critical reviews mention each:

| Segment | Example Keywords |
|---|---|
| Service / Staff | service, staff, rude, waiter, manager, slow, waiting |
| Food Quality / Taste | taste, quality, cold, stale, bland, undercooked |
| Pricing / Value | expensive, overpriced, price, money, value |
| Order Accuracy / Delivery | wrong, missing, delivery, packed, delayed |
| Ambience / Facilities | ambience, noisy, crowded, seating, parking |
| Hygiene / Cleanliness | hygiene, dirty, unclean, insect, hair, smell |

**Insight:** Service (service, bad, worst, staff) and food quality (taste, quality) dominate the top keywords. Words like "good" still surface due to negated phrases ("not good") — a known limitation of plain frequency analysis.

### Phase 7: Post-EDA Visualization
Generate a word cloud of complaint keywords, a bar chart of the top 20 complaint keywords, and a horizontal bar chart of critical-review percentage by complaint category.

**Insight:** Food Quality/Taste (33.2%) and Service/Staff (31.4%) together account for nearly two-thirds of all critical reviews; Pricing/Value (15.2%) and Order Accuracy/Delivery (10.5%) are secondary; Ambience (8.5%) and Hygiene (4.3%) are minor by comparison.

### Phase 8: Generative AI Apology Emails
Select the top 3 critical reviews — ranked by number of matched complaint segments, then review length — and pass each into a Generative AI API to draft a short (under 150 words), empathetic apology email that:
1. Directly acknowledges the specific issues raised,
2. Apologizes sincerely without being defensive,
3. States a concrete next step (escalation, investigation, or an offer to make it right),
4. Closes politely, without inventing details the review didn't mention.

**Provider:** Wired to **xAI's Grok API** (`grok-4.5`) via the OpenAI-compatible SDK (`base_url="https://api.x.ai/v1"`, key read from `XAI_API_KEY`). OpenAI or Gemini can be substituted by swapping the client/model inside `generate_apology_email`. A local template fallback keeps the notebook runnable without live credentials.

Generated emails are printed alongside their source reviews, then exported to `AI_Generated_Emails.txt`; the filtered critical reviews and complaint category summary are exported to `Critical_Restaurant_Reviews.csv` and `Complaint_Category_Summary.csv`.

## Phase 9: Key Insights

1. **A significant proportion of feedback requires attention** : ~24.4% of all reviews are 1–2 star (roughly 1 in 4 customers), making manual review impractical at scale.
2. **Service quality is the biggest driver of dissatisfaction** : Frequent terms like *service, staff, worst, rude* point to employee behavior, responsiveness, and wait time as the top pain points.
3. **Food consistency is a recurring operational challenge** :  Keywords like *taste, quality, chicken, biryani, rice* suggest inconsistent food quality on popular menu items.
4. **Complaints concentrate around a few recurring themes** rather than being random, so targeted fixes can address a large share of dissatisfaction.
5. **Manual review doesn't scale** : With 10,000 reviews, automated filtering and keyword extraction enable much faster prioritization than manual reading.

## Phase 10: Strategic Recommendations

1. **Prioritize service-quality training** : Invest in front-of-house training to reduce wait times and improve staff courtesy and attentiveness.
2. **Audit kitchen consistency** : Standardize preparation for high-complaint dishes (biryani, chicken, rice) and run regular quality checks.
3. **Operationalize the review analysis pipeline** : Run it weekly/monthly and track complaint-category trends over time via a KPI dashboard.
4. **Use AI-generated responses for faster support** : Auto draft empathetic replies for every critical review; agents review and personalize before sending.
5. **Extend analysis to restaurant/branch level** : Run keyword and category analysis per location to identify and target underperforming branches.

## Phase 11: Executive Summary, Future Scope & Conclusion

### Executive Summary
Managing thousands of customer reviews manually is inefficient and hard to scale. This project presents a Python-based automated review analysis pipeline that filters critical (1–2 star) reviews with rule-based logic, performs keyword frequency analysis to uncover complaint themes, and leverages Generative AI to draft personalized apology emails for high-priority cases — all without traditional machine learning. Across 10,000 restaurant reviews, ~24% were critical, with service quality and food/order quality emerging as the primary drivers of dissatisfaction. The result is a fast, transparent, cost-effective feedback management system that reduces manual effort, accelerates response times and gives management continuous visibility into recurring operational issues.

### 💡 Future Scope
1. **Improve keyword extraction with negation handling** (e.g., correctly interpreting "not good," "not fresh").
2. **Incorporate lightweight sentiment analysis** or topic modeling as an optional enhancement.
3. **Develop a real-time monitoring dashboard** tracking critical-review rate and complaint trends per restaurant/branch.
4. **Integrate with customer support systems** so AI-generated drafts can be reviewed, approved, and sent within one workflow.
5. **Expand to multiple feedback channels** : App store reviews, support tickets, surveys, social media — using the same filter → insight → AI-response workflow.

### 📌 Conclusion
This project demonstrates that a simple, explainable, rule-based approach combined with Generative AI only at the final response-generation stage can significantly reduce the manual effort required for customer feedback analysis and response management. It avoids the complexity of an ML training pipeline while remaining transparent, easy to implement and auditable, offering a reusable template for organizations looking to improve customer experience, accelerate response times and enhance retention through data-driven feedback management.

---

## 🛠️ Tools & Technologies

| Tool | Purpose |
|---|---|
| Python | Programming |
| Pandas / NumPy | Data loading, cleaning & numerical computing |
| Matplotlib / Seaborn | EDA visualization (rating distribution, keyword charts, category charts) |
| WordCloud | Complaint-keyword word cloud |
| `collections.Counter` | Rule-based keyword frequency analysis (no ML) |
| Grok API (OpenAI-compatible SDK) | Generative AI apology email drafting |
| Jupyter Notebook | Development environment |

---

## ▶️ How to Run

```bash
pip install pandas numpy matplotlib seaborn wordcloud openai
jupyter notebook
```

1. Place `Restaurant reviews.csv` in an accessible folder and update the file path in the **Loading the Data** cell:
   ```python
   df = pd.read_csv("path/to/Restaurant reviews.csv")
   ```
2. Set your Generative AI API key as an environment variable:
   ```bash
   export XAI_API_KEY="your-api-key-here"      # macOS/Linux
   setx XAI_API_KEY "your-api-key-here"         # Windows
   ```
3. Run the notebook cells sequentially (Phases 1–8 as documented above).
4. Review the exported `Critical_Restaurant_Reviews.csv`, `Complaint_Category_Summary.csv`, and `AI_Generated_Emails.txt` files.

---

## Summary Table

| Metric | Value |
|---|---|
| Total reviews analyzed | ~10,000 |
| Critical reviews (1–2 star) | 24.4% |
| Top complaint category | Food Quality / Taste (33.2%) |
| Second complaint category | Service / Staff (31.4%) |
| Least common complaint category | Hygiene / Cleanliness (4.3%) |
| AI apology emails generated | 3 (top critical & most detailed reviews) |
| Machine learning models used | 0 (rule-based only) |

---

## 📄 License

Copyright (c) 2026

All Rights Reserved.

This project and all associated files are provided for viewing and portfolio demonstration purposes only. No part of this project may be reproduced, copied, modified, distributed, published, sublicensed, or sold in any form without prior written permission from the copyright holder. Unauthorized use of this project is strictly prohibited.

---

## 🙋 Author
**Uday Sahu**
**Data Analytics Project | Customer Experience & Feedback Automation Domain**

