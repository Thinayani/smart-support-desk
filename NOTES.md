### Design choices and trade-offs

- FastAPI, SQLite and plain HTML: I kept the app simple and easy to run. The downside is that it uses a shared team login and has no rate limiting on the public form.
- Classifier: TF-IDF (word + char n-grams) + logistic regression. Char n-grams help with typos, and the data is easy. The app stores and shows the model's confidence; it never routes on category alone.
- TF-IDF and logistic regression: I chose this because it's lightweight and works well with the available data. Character n-grams also help with spelling mistakes. However, the dataset may not fully represent real customer messages.
- Replies are grounded by construction: Retrieval runs first and is deterministic. With no matching FAQ entry the model is never called and the draft says it has no answer. If it is called, its JSON must cite retrieved entries, promise nothing, and stay close to the FAQ wording, or a fixed FAQ-based draft is used.
- FAQ-based AI replies: I made sure the system checks the FAQ first and only uses Gemini when it finds relevant information. This helps avoid made-up answers, but sometimes the retrieval system misses questions written with typos.

### Two places the AI tool gave wrong output

1. The grouping code skipped some duplicate pairs because it incorrectly assumed the first neighbour was the original row. I caught this with a unit test using two identical messages and fixed the code.
2. The initial thresholds were not suitable. I noticed this by checking the class counts and testing simple questions like "where is my order". I adjusted the grouping threshold from 0.80 to 0.85 and the retrieval threshold from 0.30 to 0.20.

### With one more week

1. Add more tickets to the B3 evaluation set and repeat the tests to get more reliable results.
2. Improve FAQ retrieval using embeddings and a reranker.
3. Fix confusing ticket categories and create a larger, more realistic test dataset.
4. Track whether agents send, edit or reject AI-generated replies.
5. Improve security with proper authentication, password hashing and rate limiting, and move from SQLite to PostgreSQL.
