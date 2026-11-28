# Milestone 1: Proposal and Setup (Team Kramer)

**Project:** Classifying Philippine Import Goods into HS Chapters from Their Descriptions  
**Repository:** https://github.com/jauntsmirk/cmsc170-project

---

## A. Group Information

**Group Name:** Team Kramer

**Members:**
1. Mark Justin Feliciano
2. Ivan Joas Managat
3. Jose Anrhed Pesimo

---

## B. Problem Statement

Every shipment entering the Philippines is declared to the Bureau of Customs, and each declaration describes its goods in free text, such as "PALM OLEIN IN BULK" or "CABLE 6M WITH WATERPROOF CONNECTOR". The declarant also assigns the goods a tariff code, and that code determines the duty rate the shipment pays. Our dataset records about 1.2 million import transactions in 2012 and about 9.4 million in 2025.

Because the descriptions are free text, the same product can be worded many ways. Checking by hand that each description matches its declared code becomes harder as volume grows, and a wrong code can mean the wrong duty is paid.

We propose letting a model read the description and predict where the goods belong in the Harmonized System (HS), the international standard for naming and numbering traded products, maintained by the World Customs Organization. The HS covers more than 5,000 commodity groups, each with a six-digit code that is the same worldwide. The digits nest: the first two give the chapter (a broad product group, 96 in all), the first four the heading, and the first six the subheading. Countries add their own digits for national tariff lines, so codes in this dataset typically have 11 digits. For example, "PALM OLEIN IN BULK" appears under code 15119090000, so its chapter is 15 (animal and vegetable fats and oils). Every import declaration carries an HS code, so the data gives us this label at scale.

Our task is **multi-class text classification**: given the goods description on an import declaration, predict its HS chapter (the first two digits of the code). A model that does this well could help brokers and customs staff triage declarations and flag descriptions that do not match their declared code. We compare how well different AI methods learn this from data and why some do better than others, including how well each one holds up on years it was not trained on.

---

## C. Dataset

We use the Philippine Bureau of Customs import transactions for 2012 to 2025, chosen from the instructor's list of available datasets.

- **Source:** The Philippine Bureau of Customs import and export transaction data compiled by BetterGov.PH. Datasets from 2012–2025.
- **Size:** The unified dataset has 65,163,468 import transactions in 37 columns. We work from the per-year files `2012.csv` to `2025.csv`, which range from 205 MB to 1.8 GB. We will use only a sample of about 210,000 rows total, using reservoir sampling on each year's file.
- **Target:** The HS chapter, taken as the first two digits of `hs_code`. Labels are the codes importers declared, so some label noise is expected.

---

## D. Methods

We will implement and compare three methods from scratch, each from a different family. All three solve the same task on the same data.

| # | Method | Why it suits this problem |
|---|---|---|
| 1 | **Naive Bayes with Laplace smoothing** | Word counts per chapter, the same bag-of-words idea as the spam and ham lab but with about 96 classes. Smoothing handles words never seen in a chapter. |
| 2 | **k-Nearest Neighbors** | Classifies a description by the most similar training descriptions. We use cosine similarity on TF-IDF weights. |
| 3 | **Perceptron** | A linear classifier on word features. We extend the lab's binary rule to many classes with one weight vector per chapter, updated when the predicted chapter is wrong. |

---

## E. GitHub Repository

- **Repository:** https://github.com/jauntsmirk/cmsc170-project
- Repository created and instructor added as collaborator.

---

## F. References

- Bureau of Customs of the Philippines. (2012 to 2025). *Import Transaction Records.* Combined and unified dataset.
- Abante, K. (2020). *An Open Data Set of Twenty Million Import Transactions from the Bureau of Customs of the Philippines (2012–2019).* doi:10.13140/RG.2.2.24053.73444
- World Customs Organization. *What is the Harmonized System (HS)?* wcoomd.org.

---

## G. AI Tool Disclosure

We used Claude to search for and compare candidate datasets and to help draft this proposal. All choices were reviewed by the group, and all code will be written by the group.
