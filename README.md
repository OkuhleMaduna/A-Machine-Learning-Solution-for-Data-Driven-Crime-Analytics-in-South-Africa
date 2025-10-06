# A-Machine-Learning-Solution-for-Data-Driven-Crime-Analytics-in-South-Africa
South Africa faces persistent and diverse crime challenges that affect citizens, businesses, and city development
# 🧠 A Machine Learning Solution for Data-Driven Crime Analytics in South Africa

##  Student Details
**Name:** OKUHLE MADUNA  
**Student Number:** 22410169  
**Course:** Data Science and Machine Learning  

---

##  Project Overview
This project presents a **machine learning-driven dashboard** for analyzing and forecasting crime in South Africa.  
It integrates **classification** and **time series forecasting** models to identify crime hotspots and predict future trends, providing actionable insights for law enforcement and policymakers.

---

#
1. **Crime Statistics for South Africa**  
   - [Kaggle Dataset](https://www.kaggle.com/datasets/slwessels/crime-statistics-for-south-africa)  
   - Contains provincial-level crime data from 2005–2016 across categories.  
   - Columns: `Province`, `Station`, `Category`, `2005-2006` ... `2015-2016`.

2. Province Population Dataset** (Included in project ZIP)  
   - Contains demographic and density information for South African provinces.  
   - Columns: `Province`, `Population`, `Area`, `Density`.

---

Features of the Streamlit Dashboard
The dashboard provides the following functionalities:

 Home
- Displays project title, author, purpose, and key tasks.
 EDA & Filtering
- Allows dataset upload.
- Filters by **crime category**, **province**, and **year**.
- Displays interactive visualizations of crime statistics.

 Classification
- Uses **Random Forest Classifier** to identify crime hotspots.
- Automatically computes **model accuracy** and shows a **confusion matrix**
-  Forecasting
- Uses **Facebook Prophet** for 24-month **crime trend forecasting**.
- Displays forecast plots with confidence intervals.

 Settngs
- Allows theme switching between **Light Mode** and **Dark Mode**.

--- Running the Application
Requirements
Install required dependencies:
```bash
pip install streamlit prophet scikit-learn pandas matplotlib
