# Dynamic Trend & Event Detector: Presentation Speech Script

*This script is designed to match your project, the report, and the likely flow of your presentation (Semantic Radar / Dynamic Trend Detector). It includes cues for when to switch slides and how to explain the graphs.*

---

## Slide 1: Title Slide
**Speech:**
"Good morning everyone. My name is Navnit Naman, and along with my partner, Kanhaiya Kumar, we are excited to present our Project: the **Dynamic Trend & Event Detector**, which we also like to call our 'Semantic Radar.' Today, we will be walking you through a deep semantic evolution framework that we built to monitor societal narratives and detect breaking global events in real-time."

---

## Slide 2: Introduction & Problem Statement
**Speech:**
"Let me start by defining the problem. In today’s world, information streams like news and social media are incredibly fast and overwhelming. Traditional systems try to track trends just by counting keywords. But they miss the 'fluidity' of meaning—for example, a keyword counter doesn't know that 'Price Hike' and 'Inflation' are talking about the exact same problem. Our challenge was to build a system that can distinguish between everyday, routine news reporting and sudden, structural societal ruptures. Our project solves this for domains like social media analytics, journalism, and policy-making."

---

## Slide 3: Dataset Quality & Exploratory Data Analysis (EDA)
*(If your slide shows the EDA temporal distribution and Top Themes graphs)*

**Speech:**
"To build this, we ingested two massive datasets: The ABC News Headlines dataset and the real-time GDELT Global Knowledge Graph. Looking at our exploratory data analysis on the screen, you can see the temporal distribution of headlines and the frequency of top global themes. We discovered that news narratives are highly non-stationary—topics burst onto the scene abruptly rather than growing smoothly, which confirmed that we needed a dynamic, machine-learning approach rather than simple statistics."

---

## Slide 4: Layered Implementation Strategy (The 3 Paths)
**Speech:**
"We designed a three-layered architecture. First, a **Baseline** using Statistical TF-IDF for fast keyword tracking. Second, an **Advanced Machine Learning** layer using Probabilistic LDA for thematic discovery. And third, our **Deep Learning** layer which uses Transformer-based models, specifically BERTopic, for high-precision clustering. We implemented all three so we could rigorously compare traditional methods against modern semantic AI."

---

## Slide 5: Baseline & Advanced ML (TF-IDF & LDA)
*(If your slide shows the Baseline Burst Graph and LDA Topic Distribution Graph)*

**Speech:**
"Here you can see the results of our first two layers. The Baseline graph tracks keyword bursts over time, alerting us when specific words suddenly dominate the news. The second graph shows our LDA topic distribution. While LDA is an industry standard, we found it suffers from 'Topic Smearing' on short texts like headlines—it groups unrelated headlines together just because they share common nouns. This limitation proved exactly why we needed our deep learning layer."

---

## Slide 6: Deep Learning (BERTopic & Semantic Clusters)
*(If your slide shows the BERTopic UMAP/HDBSCAN Cluster Graph)*

**Speech:**
"To solve the LDA sparsity problem, we implemented BERTopic using Sentence-BERT embeddings. Instead of counting words, we project entire sentences into a high-dimensional semantic space. As you can see in our cluster projection graph, we used UMAP for dimensionality reduction and HDBSCAN for density clustering. This allowed us to visually and mathematically separate entirely distinct global narratives with incredible precision, safely isolating background noise."

---

## Slide 7: The Innovation: Semantic Velocity ($V_s$)
*(If your slide shows the Semantic Velocity line chart)*

**Speech:**
"This brings us to one of our core mathematical innovations: **Semantic Velocity**, or $V_s$. Instead of just identifying topics, we calculate how fast a narrative is moving. We do this by measuring the Cosine Distance between the dimensional centroids of consecutive time-buckets. Looking at the semantic velocity graph, the massive spikes you see represent 'Narrative Ruptures'—these are exact moments when a breaking geopolitical or societal event drastically altered the global conversation."

---

## Slide 8: Real-Time Sensing & Event Impact Scoring
*(If your slide shows the Event Impact Distribution Graph)*

**Speech:**
"Detecting a trend isn't enough; analysts need to know its impact. To go the extra mile, we integrated the GDELT real-time stream and created the **Event Impact Score**, or $S_I$. We calculate this by multiplying the 'Semantic Uniqueness' of a cluster by its emotional intensity, or Tone. The distribution graph shows how this isolates the signal from the noise: a standard scheduled meeting gets a very low impact score, but a unique, emotionally heavy breaking news event gets a massive spike, instantly alerting journalists or policymakers."

---

## Slide 9: Ethics & Misinformation Detection
**Speech:**
"With great analytical power comes ethical responsibility. We deliberately chose to integrate GDELT because it sources from over 100 languages, actively mitigating the western-bias often found in pre-trained BERT models. Furthermore, our Semantic Velocity metrics double as a misinformation detector. When we see high velocity paired with highly repetitive, narrow clusters in short time windows, it strongly flags bot-driven 'Echo Chambers' and coordinated influence operations."

---

## Slide 10: Conclusion
**Speech:**
"To conclude, our Dynamic Trend and Event Detector successfully bridges the gap between traditional statistical rigor and neural context. By engineering custom metrics like Semantic Velocity and Event Impact Scoring, we have built a scalable, industrial-ready solution for proactive societal monitoring. 

Thank you for your time. We are now open to any questions you might have about our architecture, mathematical formulas, or findings."
