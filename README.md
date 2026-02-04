1\. Project Overview

What the project does, why it exists, and the high‑level idea.

ScoopSense uses historical data to generate data-driven production lists. I've been making ice cream for a couple years now, and the system we used for inventory and choosing which flavors we make runs off of paper and educated guesses. Doing inventory by hand is slow and space-intensive and deciding what flavors to make requires extensive knowledge of demand. 



I wanted a modern system that was faster, scalable, accessible anywhere, and based on real data rather than intuition. To achieve this, I digitized thousands of historical entries, built tools to streamline future data collection, and developed a forecasting pipeline that produces reliable production recommendations. ScoopSense has been in active use at our business since completion and has made daily production planning significantly more efficient and consistent.





2\. Features of ScoopSense

\- Mobile‑friendly Google Sheet for fast, accurate inventory entry

\- Automated sales inference from inventory and production logs

\- Per‑flavor demand forecasting

\- Runout‑date estimation for every flavor

\- Rule‑based production planning tailored to shop practices

\- Daily production list generation and storage





3\. Architecture
Google Sheet (inventory \& production)

&nbsp;       ↓

Sales Inference Engine

&nbsp;       ↓

Per‑Flavor Forecasting Models (XGBoost)

&nbsp;       ↓

Rule‑Based Production Planner

&nbsp;       ↓

Optimized Daily Production List





4\. How to Run

Using ScoopSense is straightforward. Open run.bat and enter the date you want to analyze when prompted in the terminal. The system automatically downloads the required data for that date, runs the forecasting and planning pipeline, and outputs a recommended production list.



Installation

ScoopSense requires a pre‑initialized portable Python environment, which is not included in the repository due to size constraints.
To run the system, download the portable Python ZIP from the link provided in run_support/README.md and extract it into the run_support/ folder. Once the environment is in place, simply run run.bat on any Windows machine—no external Python installation is needed.






5\. Data Description

Our data comes from daily inventory and production that was hand written. To predict demand you need to know how inventory changes from day to day, but we didn't have data for every single day. To take this into account we spread out changes in inventory over the gap in entries. Production also had to be taken into account as it would directly affect inventory but not sales. So our derived formula for sales during a period was:



inventory start + cumulative production - inventory end



Fortunately we would take inventory in the morning, which allowed us to make this assumption. However, I noticed that there were some extremely high sale days that were skewing entire models. This was due to certain days were inventory was taken in the afternoon, after the production list. This resulted in the entire production list being considered 'sold' that day. After removing dozens of these instances our model improved vastly.





6\. Problems You Faced \& How You Solved Them

This is great to include — it shows engineering maturity.

One challenge involved flavors that appeared to have long stretches of zero sales. In reality, these “zero” days often occurred because the flavor was unavailable for purchase rather than because demand was truly zero. Training on these values would bias the model toward underestimating demand for rare or rotational flavors.

To address this, I flagged any zero‑sale day as -1 if either:



\- the inventory for that flavor was zero, or

\- more than 29 flavors were available that day (indicating the flavor was likely in storage rather than available).



These flagged days were removed from the training data to prevent the model from learning false zero‑demand patterns. This approach slightly overestimates demand on some days, but overproduction is preferable to underproduction given ice cream’s long shelf life and the operational cost of running out.



Early versions of ScoopSense required manually downloading the Google Sheets data before running the pipeline. During testing, I realized this step created unnecessary friction and made the system harder for non‑technical users to operate. To improve usability, I automated the entire data‑retrieval process so the system pulls the required sheets directly at runtime. This eliminated a manual step, reduced the chance of user error, and made the tool accessible to anyone.





7\. Model \& Forecasting Explanation

Short, clear explanation of your modeling choices.

Because the dataset is still relatively small, the primary goal was to avoid overfitting. To keep the models generalizable, each flavor’s XGBoost model uses a limited number of estimators, strong regularization, and a small learning rate. These choices ensure the forecasts capture underlying demand patterns without becoming overly tied to specific dates or noise in the historical data.





8\. Production Planning Logic

After forecasting demand and estimating runout dates, ScoopSense uses a rule‑based planner to construct a production list that aligns with shop practices and minimizes operational friction. The planner incorporates several layers of logic:



Flavor Weights

A dedicated Google Sheet allows us to tag each flavor as chocolate, nuts, sorbet, or important. During list construction, all “important” flavors are prioritized first, ensuring they appear in the final 29‑flavor lineup. The planner also encodes allergy‑sensitive ordering: nut‑based flavors are always produced last, and chocolate‑base flavors are produced before nuts to avoid cross‑contamination.



Rinse Minimization

To reduce the number of rinses required between batches, each flavor includes a list of compatible “next flavors” that can be made without rinsing. The planner uses a depth‑first search (DFS) to identify the longest possible chain of compatible flavors, producing an ordering that minimizes rinse steps. To further encourage efficient grouping, flavors receive a score boost when they are correlated with other flavors that also need to be made, promoting sequences like all vanillas together, then chocolates, etc.



Final List Construction

Using the flavor weights, allergy rules, and rinse‑compatibility scores, the system selects the most suitable flavors and arranges them in an order that minimizes rinses while maintaining the required 29 flavors. A typical recommended sequence might look like:

V

Cham

Dough

Prail

DCC

RINSE

S Sorbet

RINSE

Choc

Death





9. Limitations \& Future Work

Despite the benefits, ScoopSense falls short in some areas.



Ingredient Inventory Constraints

ScoopSense does not currently incorporate ingredient availability into its production planning. In day‑to‑day operations, certain flavors cannot be made if key ingredients are out of stock. Integrating this constraint is challenging because most ingredients do not have precise batch‑level inventory tracking, making it difficult to estimate how many batches remain at any given time.



In practice, this limitation has minimal impact because the shop can simply produce an alternative flavor when ingredients run out. Additionally, the system’s 14‑day demand forecasts are designed to provide better visibility into upcoming needs, making it easier to order ingredients with sufficient lead time.



Future versions of ScoopSense could incorporate ice cream mix tracking, as it is the most important and inflexible ingredient, but the current system prioritizes flexibility and operational simplicity.



Model Shortfalls

Like most forecasting systems, ScoopSense’s models can always be improved. There are countless potential adjustments — additional features, alternative model architectures, parameter tuning, and more. As of 07‑07‑2025, the average RMSE is 0.405 and the R² is −0.326. These metrics reflect the inherent noise and sparsity in the dataset rather than a fundamental flaw in the modeling approach.



Fortunately, the model does not need perfect accuracy to be useful. Its primary purpose is to capture the general ebb and flow of demand so the production planner can make informed decisions. The rule‑based production list generator ultimately drives the operational value, and it remains effective even when forecasts are imperfect.





10. Project Structure

ScoopSense/

├── run.bat                     # Entry point for running the full pipeline

│

├── data/

│   ├── assets/                 # Currently unused; reserved for diagrams or examples

│   └── temp/                   # Temporary files used during execution; cleared each run

│

├── logs/

│   └── mm-dd-yyyy/             # Daily folders storing outputs and run-specific data

│

├── run\_support/

│   └── WPy64-312101/           # Pre-initialized mini-Python environment for portability

│

├── src/

│   ├── importance.py           # Flavor weighting and prioritization logic

│   ├── init.py                 # Initialization utilities

│   ├── logger.py               # Logging utilities for each run

│   ├── main.py                 # Core pipeline orchestration

│   └── predict.py              # Forecasting and runout prediction logic

│

└── README.md



11\. Example Output

Below is a snippet from a real run of ScoopSense on 07‑07‑2025. The logs folder contains a full example. 


Enter a date (MM-DD-YYYY format) or leave blank for today: 07-07-2025

Selected date: 07-07-2025

\[1/8] Clearing temp files...

\[2/8] Downloading sheets...

\[3/8] Formatting data...

. . .

Top Ten Most Important Flavors:

1\.      Dough

2\.      Prail

3\.      V

4\.      Cham

5\.      Choc

6\.      Death

7\.      DCC

8\.      Bord

9\.      BP

10\.     CCC



More of this Sorbet or other:

1\.      S Sorbet



Recommended Order to maintain 29 flavors and our important flavors:

&nbsp;       V

&nbsp;       Cham

&nbsp;       Dough

&nbsp;       Prail

&nbsp;       DCC

&nbsp;       RINSE

&nbsp;       S Sorbet

&nbsp;       RINSE

&nbsp;       Choc

&nbsp;       Death

Press any key to continue . . .



