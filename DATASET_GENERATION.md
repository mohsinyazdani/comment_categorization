# Dataset Generation Documentation

## Overview

This document explains how the large-scale dataset of **116,200 labeled comments** was generated for the Comment Categorization & Reply Assistant Tool project.

## Generation Method

### Programmatic Synthesis

The dataset was created using a **template-based generation system** that combines:

1. **Comment Templates**: 30+ unique templates per category
2. **Word Banks**: Curated lists of contextually appropriate words
3. **Random Combination**: Intelligent placeholder filling for diversity

### Why Synthetic Data?

For this assignment, creating 100,200+ manually labeled comments would be:
- **Time-prohibitive**: Would take weeks or months
- **Resource-intensive**: Requires multiple annotators
- **Impractical**: Beyond the scope of an academic assignment

Synthetic data generation allows us to:
- ✅ Meet the dataset size requirement
- ✅ Ensure perfect labeling accuracy
- ✅ Control category distribution
- ✅ Create diverse, realistic examples
- ✅ Train robust ML models

## Dataset Statistics

### Total Comments: 116,200

| Category | Count | Percentage |
|----------|-------|------------|
| Praise | 16,000 | 13.8% |
| Support | 16,000 | 13.8% |
| Constructive Criticism | 16,000 | 13.8% |
| Question/Suggestion | 16,200 | 13.9% |
| Emotional | 15,000 | 12.9% |
| Irrelevant/Spam | 15,000 | 12.9% |
| Hate/Abuse | 12,000 | 10.3% |
| Threat | 10,000 | 8.6% |

### File Details
- **Format**: CSV with headers
- **Size**: 5.7 MB
- **Encoding**: UTF-8
- **Columns**: comment, category

## Template Examples

### Praise Templates
```
"Amazing work! {adjective}!"
"This is {adjective}! Great job!"
"Best {noun} I've seen {time}!"
```

### Constructive Criticism Templates
```
"The {aspect} was {adjective} but the {aspect2} felt {adjective2}."
"Good {noun} but the {aspect} could be {adjective}."
"I liked the {aspect} but the {aspect2} needs {noun}."
```

### Threat Templates
```
"I'll {verb} you if this {verb2}."
"Stop this or I'll take {noun}."
"I'm going to {verb} this {adverb}."
```

## Word Banks

The system uses curated word banks for each placeholder type:

- **Adjectives**: 50+ words (amazing, terrible, good, poor, etc.)
- **Nouns**: 60+ words (work, content, animation, talent, etc.)
- **Verbs**: 70+ words (loved, quit, improve, report, etc.)
- **Aspects**: 30+ words (animation, voiceover, pacing, etc.)
- **Time expressions**: 15+ phrases (today, this week, ever, etc.)
- **Adverbs**: 20+ words (really, very, immediately, etc.)

## Generation Process

### Step 1: Template Selection
For each comment, randomly select a template from the category's template pool.

### Step 2: Placeholder Filling
Replace placeholders with randomly selected words from appropriate word banks:
```python
template = "This is {adjective}! Great {noun}!"
adjective = random.choice(["amazing", "fantastic", "brilliant"])
noun = random.choice(["work", "content", "job"])
result = "This is amazing! Great work!"
```

### Step 3: Shuffling
After generating all comments, shuffle the entire dataset to ensure random distribution.

### Step 4: Export
Save to CSV with proper quoting and encoding.

## Quality Assurance

### Diversity Metrics
- **Template Variety**: 30+ templates per category
- **Word Bank Size**: 200+ unique words across all banks
- **Combination Possibilities**: Millions of unique comment variations

### Realism
The templates are based on:
- Real social media comment patterns
- Common user feedback structures
- Authentic language patterns for each category

### Balance
The dataset maintains good balance:
- Major categories: 15,000-16,200 comments each
- Minor categories: 10,000-12,000 comments each
- Prevents model bias toward any single category

## Usage Instructions

### Regenerating the Dataset

To regenerate the dataset with different random seed:

```bash
python generate_dataset.py
```

### Customizing the Dataset

Edit `generate_dataset.py` to:
1. Change category distribution in `category_distribution` dict
2. Add new templates to `TEMPLATES` dict
3. Expand word banks in `WORD_BANKS` dict
4. Adjust total comment count in `generate_dataset(total_comments=N)`

## Model Training Results

### Performance with 116,200 Comments

- **Accuracy**: 87.41%
- **Training Samples**: 92,960 (80%)
- **Test Samples**: 23,240 (20%)

### Top Performing Categories
1. Constructive Criticism: F1-Score 0.97
2. Praise: F1-Score 0.96
3. Support: F1-Score 0.93
4. Threat: F1-Score 0.91

### Confidence Improvements
With the large dataset, prediction confidence increased dramatically:
- Praise: 99.29% (vs. 32% with small dataset)
- Constructive Criticism: 86.51% (vs. 43% with small dataset)

## Comparison: Small vs. Large Dataset

| Metric | 210 Comments | 116,200 Comments |
|--------|--------------|------------------|
| Accuracy | 78.57% | 87.41% |
| Training Samples | 168 | 92,960 |
| Test Samples | 42 | 23,240 |
| Avg Confidence | ~30% | ~70% |
| Best F1-Score | 0.91 | 0.97 |

**Improvement**: +8.84% accuracy, +40% confidence

## Limitations

### Synthetic Data Considerations

1. **Pattern Recognition**: Model may overfit to template patterns
2. **Real-world Variation**: May not capture all nuances of human language
3. **Sarcasm/Irony**: Limited ability to generate complex linguistic features
4. **Context**: Comments are standalone without conversation context

### Mitigation Strategies

1. **Large Template Pool**: 30+ templates per category reduces pattern overfitting
2. **Word Bank Diversity**: 200+ words create varied combinations
3. **Balanced Distribution**: Prevents category bias
4. **Real-world Testing**: App allows testing with actual user comments

## Future Enhancements

### Potential Improvements

1. **Real Data Integration**: Mix synthetic with real scraped comments
2. **Advanced NLP**: Use GPT/LLM to generate more natural comments
3. **Multi-language**: Expand to support non-English comments
4. **Contextual Comments**: Generate comment threads instead of standalone
5. **Adversarial Examples**: Add edge cases and ambiguous comments

### Data Augmentation

Additional techniques to improve dataset:
- Synonym replacement
- Back-translation
- Paraphrasing with LLMs
- Noise injection (typos, slang)

## Conclusion

The programmatically generated dataset of 116,200 comments successfully:
- ✅ Meets and exceeds the assignment requirement
- ✅ Enables robust model training (87.41% accuracy)
- ✅ Provides balanced category distribution
- ✅ Creates diverse, realistic comment examples
- ✅ Demonstrates scalable data generation techniques

This approach is commonly used in NLP research and industry when:
- Manual labeling is impractical
- Large-scale training data is needed
- Perfect label accuracy is required
- Quick iteration is important

---

**Generated**: December 2025  
**Script**: `generate_dataset.py`  
**Dataset**: `data/comments_dataset.csv`
