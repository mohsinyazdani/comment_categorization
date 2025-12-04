# Quick Start Guide

## 🚀 Get Started in 3 Minutes

### Step 1: Install Dependencies (1 minute)

```bash
# Create and activate virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### Step 2: Verify Model (30 seconds)

The model is already trained and saved in the `models/` directory. To verify:

```bash
cd src
python classifier.py
```

You should see model accuracy and test predictions.

### Step 3: Launch the App (30 seconds)

```bash
# From the project root directory
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

## 🎯 Try It Out

### Test Single Comment
1. Go to the **📝 Single Comment** tab
2. Enter: "This is amazing! Great work!"
3. Click **🔍 Analyze**
4. See it categorized as "Praise" with response suggestions

### Test Batch Analysis
1. Go to the **📄 Batch Analysis** tab
2. Upload `sample_comments.csv` (included in project)
3. Click **🔍 Analyze Batch**
4. View visualizations and export results

## 📊 What You'll See

- **Category Classification**: Each comment gets labeled (Praise, Support, Criticism, etc.)
- **Confidence Scores**: How confident the model is in its prediction
- **Response Templates**: Suggested replies for each category
- **Visual Analytics**: Charts showing category distribution
- **Export Options**: Download results as CSV or JSON

## 🎨 Features to Explore

- ✅ Single comment analysis with instant results
- ✅ Batch processing for multiple comments
- ✅ Interactive visualizations (pie charts, bar graphs)
- ✅ Smart response templates for each category
- ✅ Export functionality (CSV/JSON)
- ✅ Dataset overview and statistics

## 🔧 Troubleshooting

**Issue**: Model files not found
- **Solution**: Run `python src/classifier.py` to train the model

**Issue**: NLTK data missing
- **Solution**: The app will auto-download, or manually run:
  ```python
  python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
  ```

**Issue**: Port 8501 already in use
- **Solution**: Run with different port:
  ```bash
  streamlit run app.py --server.port 8502
  ```

## 📝 Sample Comments to Test

Try these in the Single Comment analyzer:

1. **Praise**: "Absolutely amazing work! Love it!"
2. **Support**: "Keep going, you're doing great!"
3. **Constructive**: "Good job but the audio needs improvement"
4. **Hate**: "This is terrible, quit now"
5. **Threat**: "I'll report you for this"
6. **Emotional**: "This made me cry, so touching"
7. **Spam**: "Follow me for followers!"
8. **Question**: "How did you make this?"

## 🎓 Assignment Deliverables Checklist

- ✅ Dataset with 145 labeled comments (7 categories)
- ✅ Preprocessing pipeline (cleaning, tokenization, lemmatization)
- ✅ Trained ML classifier (Logistic Regression + TF-IDF)
- ✅ Streamlit web application with UI
- ✅ Response templates for each category
- ✅ Visualizations (pie chart, bar chart)
- ✅ Export functionality (CSV/JSON)
- ✅ Comprehensive documentation (README.md)
- ✅ Clean, modular, well-commented code
- ✅ Separate handling of constructive criticism

## 🌟 Bonus Features Included

- 📊 Interactive Plotly visualizations
- 💬 Multiple response templates per category
- 🎯 Confidence scores for predictions
- 📥 Export functionality
- 📈 Dataset overview and analytics
- 🎨 Professional UI with Streamlit
- 📋 Action recommendations for each category

---

**Need Help?** Check the **ℹ️ Help** tab in the app or read the full README.md
