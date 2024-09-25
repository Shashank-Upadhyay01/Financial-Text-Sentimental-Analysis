# Financial-Text-Sentimental-Analysis
This repository implements a financial text sentiment analysis project using Python. It offers two approaches to gather news data and analyze sentiment:

Method 1: Utilizing Yahoo Finance RSS Feeds

Leverages the pre-trained FinBERT model from ProsusAI for sentiment analysis.
Efficiently retrieves relevant news articles by parsing Yahoo Finance RSS feeds for a specific stock ticker symbol.
Focuses on keywords within the summaries to filter articles related to the chosen stock.
Calculates an overall sentiment score (positive, negative, or neutral) based on individual article sentiment.
Method 2: Employing the News API

Employs the FinBERT model for accurate sentiment analysis in the financial domain.
Retrieves news articles related to a specified keyword using the News API.
Filters articles based on the keyword presence in the title or description (case-insensitive).
Analyzes the sentiment of the article content using the FinBERT model.
Calculates an overall sentiment score based on individual article sentiment.
Key Features:

FinBERT Integration: Utilizes the FinBERT model for accurate financial sentiment analysis.
Data Source Flexibility: Provides two methods for gathering news data (Yahoo Finance RSS and News API).
Keyword Filtering: Allows for targeted analysis by focusing on articles containing a specific keyword.
Sentiment Score Calculation: Calculates an overall sentiment score for a comprehensive understanding.
Potential Applications:

Investment Decision Making: Assist investors in understanding market sentiment related to specific stocks or topics.
Risk Management: Identify potential risks based on negative sentiment trends in financial news.
Trading Strategy Development: Facilitate the development of trading strategies that incorporate sentiment analysis.
Technologies Used:

Python
feedparser (for Yahoo Finance RSS)
transformers library (for FinBERT model)
requests (for News API)
Yahoo Finance RSS feeds (optional)
News API
Future Enhancements:

Visualization Integration: Add libraries like Matplotlib or Seaborn to visualize the sentiment distribution or trends over time.
Expanded Data Sources: Explore incorporating additional news sources beyond Yahoo Finance RSS and the News API.
Comparative Analysis: Enable comparison of sentiment between different methods or keywords.
This project demonstrates the versatility of sentiment analysis in finance using FinBERT. It offers a valuable foundation for further development and customization based on your specific needs.
