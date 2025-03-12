## **📈 Stock Market Headline Attribution**
This project analyzes **financial news headlines** to determine their impact on **S&P 500 price movements**. By assigning **macro factors** to daily headlines, we can track which themes are driving market sentiment over time.

---

## **📌 Key Features**
✅ **Extract** daily market headlines and SPX returns  
✅ **Classify** headlines into macro factors  
✅ **Attribute** daily market movement to macro themes  
✅ **Compute** cumulative attributions over time  
✅ **Visualize** dominating factors over time  

---

## **📊 How It Works**
### **1️⃣ Headlines & Market Data**
- The model ingests **daily financial news headlines** and **SPX performance data**.
- A sample dataset is provided in `data/` (or can be collected from APIs).
  
### **2️⃣ Macro Factor Classification**
- Headlines are assigned to one of six macroeconomic categories based on **keyword matching**:
  - **📉 Fed Policy:** "Fed", "interest rates", "hike", "cut"
  - **💰 Earnings:** "earnings", "profit", "quarterly", "revenue"
  - **🌍 Macro:** "economy", "growth", "recession", "slowdown"
  - **⚔️ Geopolitics:** "war", "conflict", "tensions", "sanctions"
  - **📡 Tech Sentiment:** "tech", "AI", "software", "NASDAQ"
  - **🚀 Trump-Tariffs/Immigration/DOGE:** "Trump", "tariffs", "immigration", "dogecoin", "trade war"

### **3️⃣ Cumulative Sum Analysis**
- The number of days each factor is attributed to market moves is summed over time.
- This creates a **trend visualization** showing which macro themes have been dominant.

---

## **📂 Project Structure**
```
📂 stock-market-headline-attribution
│── 📜 README.md            # Project overview & instructions
│── 📜 LICENSE              # License file (MIT)
│── 📜 .gitignore           # Ignore unnecessary files (.DS_Store, .ipynb_checkpoints)
│── 📜 requirements.txt     # Python dependencies (pandas, numpy, nltk, matplotlib)
│── 📂 data                 # Example dataset (headlines.csv, spx_returns.csv)
│── 📂 notebooks            # Jupyter notebooks for analysis & visualization
│── 📂 src                  # Python scripts for data processing & modeling
│   ├── attribution_model.py   # Core model for factor classification
│   ├── utils.py               # Helper functions for text processing
│── 📂 reports              # Processed outputs & results
│── 📂 visualizations       # Charts & graphs for macro factor trends
```

---

## **🛠️ Installation**
1️⃣ **Clone the repository:**
```sh
git clone https://github.com/yourusername/stock-market-headline-attribution.git
cd stock-market-headline-attribution
```

2️⃣ **Create a virtual environment & install dependencies:**
```sh
python -m venv env
source env/bin/activate   # On Windows: env\Scripts\activate
pip install -r requirements.txt
```

3️⃣ **Run the script to process headlines:**
```sh
python src/attribution_model.py
```

---

## **🖥️ Running the Model**
- Execute the Python script to process headlines and classify macro factors:
```sh
python src/attribution_model.py
```
- Or explore the results interactively in a Jupyter Notebook:
```sh
jupyter notebook notebooks/analysis.ipynb
```

---

## **🖼️ Sample Output**
The following plot **visualizes the cumulative attribution** of macro factors driving market sentiment:

![Cumulative Attribution Graph](visualizations/cumulative_attribution.png)

---

## **📬 API & Data Sources**
📢 The project can be extended to automatically fetch headlines from APIs like:  
🔹 **Bloomberg API** – Premium market headlines  
🔹 **Yahoo Finance API** – Public news and financial data  
🔹 **NewsAPI.org** – Aggregated global financial news  

---

## **🔍 Potential Improvements**
🚀 **Use Sentiment Analysis** – Weigh headlines by positive/negative sentiment  
📈 **Machine Learning Classifier** – Train an NLP model (BERT, Naïve Bayes)  
📊 **Live Dashboard** – Deploy interactive tracking with **Streamlit** or **Flask**  

---

## **⚖️ License**
This project is licensed under the **MIT License**.  
📜 **[Read full license here](LICENSE)**.

---

## **📬 Contact & Contributions**
💡 **Author:** Steven Fandozzi  
📧 Email: [your.email@example.com](mailto:your.email@example.com)  
🔗 **GitHub:** [yourusername](https://github.com/yourusername)  

👥 **Want to contribute?** Fork the repo and submit a pull request! 🚀  
---

### **⭐ If you found this useful, consider giving it a star on GitHub! ⭐**  

---

This **README.md** will be **well-formatted and visually structured** on GitHub, making it **easy to read** and **professional-looking**. Let me know if you need modifications or additional sections! 🚀
