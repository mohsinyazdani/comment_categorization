"""
Generate a large-scale dataset of 100,200 labeled comments
This script creates diverse synthetic comments across 7 categories
"""

import pandas as pd
import random
import csv

# Set random seed for reproducibility
random.seed(42)

# Template patterns for each category
TEMPLATES = {
    "Praise": [
        "Amazing work! {adjective}!",
        "This is {adjective}! Great job!",
        "{adjective} content! Keep it up!",
        "Best {noun} I've seen {time}!",
        "{adjective} work! Really {verb}.",
        "Wow! This is {adjective}!",
        "Absolutely {adjective}! Well done!",
        "This is pure {noun}! Amazing!",
        "{adjective} work! Loved {aspect}.",
        "{adjective} quality! Really {verb} it.",
        "Excellent job! Very {adjective}.",
        "This is {adjective}! Great work!",
        "Beautiful work! Really {verb} it.",
        "{adjective}! This made my {time}.",
        "{adjective}! Absolutely {verb} it.",
        "Perfect! Couldn't be {adjective}.",
        "{adjective} work! Truly {adjective2}.",
        "Brilliant! This is {adjective}.",
        "Wonderful! Really {verb} this.",
        "{adjective} work! Very {adjective2}.",
        "This is exactly what I {verb}! Thank you!",
        "{adjective}! You're so talented!",
        "{adjective} execution! Love {aspect}!",
        "This deserves all the {noun}!",
        "You've {verb} yourself! {adjective}!",
        "Best {noun} out there!",
        "This is {adjective}! Absolutely {adjective2}!",
        "Can't stop {verb} this! So {adjective}!",
        "You're a {noun}! This is {adjective}!",
        "This made my whole {time}! Thank you!"
    ],
    
    "Support": [
        "Keep going! You're doing {adjective}!",
        "Don't give up! You've got {noun}!",
        "Stay {adjective}! We believe in you!",
        "You're on the right {noun}! Keep it up!",
        "{verb} for you! Keep {verb2}!",
        "You can do it! Don't {verb}!",
        "We're all {verb} you! Keep going!",
        "Your {noun} shows! Keep at it!",
        "You're {verb}! Keep {verb2}!",
        "Stay {adjective}! You're doing {adjective2}!",
        "Keep the {noun} going!",
        "You're making {noun}! Keep it up!",
        "Don't lose {noun}! You're doing great!",
        "Your {noun} is {verb}!",
        "Keep {verb} forward! You've got this!",
        "We {verb} you! Keep going!",
        "You're doing {adjective}! Don't stop!",
        "Keep the {adjective} work coming!",
        "Your {noun} is {adjective}!",
        "Stay {adjective}! You're doing well!",
        "Never give up on your {noun}!",
        "You're {adjective} than you think!",
        "Keep {verb} in yourself!",
        "Your {noun} is inspiring us all!",
        "We're {adjective} of your progress!",
        "Keep {verb}! You're doing great!",
        "Your {noun} is {adjective}!",
        "Stay {adjective}! You're almost there!",
        "Keep up the {adjective} effort!",
        "You're an {noun} to many!"
    ],
    
    "Constructive Criticism": [
        "The {aspect} was {adjective} but the {aspect2} felt {adjective2}.",
        "Good {noun} but the {aspect} could be {adjective}.",
        "I liked the {aspect} but the {aspect2} needs {noun}.",
        "Nice {noun} but the {aspect} is {adjective}.",
        "The {aspect} are {adjective} but the {aspect2} is {adjective2}.",
        "{adjective} idea but needs more {noun}.",
        "Good {noun} but the {aspect} felt {adjective}.",
        "I appreciate the {noun} but the {aspect} are too {adjective}.",
        "The {aspect} is {adjective} but the {aspect2} could {verb}.",
        "Nice work but the {aspect} are {adjective}.",
        "I like it but the {aspect} is too {adjective}.",
        "Good {noun} but the {aspect2} is {adjective}.",
        "The {noun} is {adjective} but needs better {aspect}.",
        "I {verb} it but the {aspect} is hard to {verb2}.",
        "Good {noun} but the {aspect} could be {adjective}.",
        "Nice {noun} but the {aspect2} is too {adjective}.",
        "I liked it but the {aspect} feels {adjective}.",
        "Good work but the {aspect} could be {adjective}.",
        "{adjective} but the {aspect2} is too {adjective2}.",
        "{adjective} but needs more {adjective2} {aspect}.",
        "The {aspect} is good but {aspect2} needs {noun}.",
        "I appreciate the {noun} but the {aspect} mixing is {adjective}.",
        "Good {noun} but could use more {aspect}.",
        "Nice {noun} but the {aspect} need {noun2}.",
        "{adjective} concept but the {aspect} is {adjective2}.",
        "I like the {noun} but needs better {aspect}.",
        "Good {noun} but the {aspect2} is too {adjective}.",
        "{adjective} approach but could be more {adjective2}.",
        "The {noun} is visible but needs {adjective} {noun2}.",
        "Good {noun} but the {aspect} lacks {noun2}."
    ],
    
    "Hate/Abuse": [
        "This is {adjective}. {verb} now.",
        "You're {adjective} at this. {verb}.",
        "This is {noun}. {verb} it.",
        "{adjective} thing I've ever {verb}. {adjective2}.",
        "You have no {noun}. {verb}.",
        "This is {adjective}. You should be {verb}.",
        "Complete {noun}. {adjective}.",
        "You're a {noun}. This is {adjective}.",
        "This {verb}. You're {adjective}.",
        "{adjective} work. You're {adjective2}.",
        "This is {noun}. Stop {verb} yourself.",
        "You're {adjective}. This is {noun}.",
        "Absolute {noun}. You have no {noun2}.",
        "This is {adjective}. You're a {noun}.",
        "{adjective}. You should {verb}.",
        "What a {noun}. {adjective}.",
        "You're the {adjective} {noun} ever.",
        "This is beyond {adjective}. {adjective2}.",
        "Nobody wants to {verb} this {noun}.",
        "You're {verb} the {noun} with this {noun2}.",
        "This is {adjective} and {adjective2}.",
        "You should {verb} making this {noun}.",
        "{adjective} content. Total {noun}.",
        "This is {adjective}. Complete {noun}.",
        "You're {adjective}. This is {noun}.",
        "What {noun}. You're {adjective}.",
        "This is {adjective}. You're a {noun}.",
        "{adjective}. Nobody {verb} this.",
        "You're {verb} everyone's {noun}.",
        "This is the {adjective} {noun} ever."
    ],
    
    "Threat": [
        "I'll {verb} you if this {verb2}.",
        "{verb} this or I'll take {noun}.",
        "I'll make sure you {verb} this.",
        "You better {verb} or {noun}.",
        "I'm going to {verb} this {adverb}.",
        "I'll get you {verb} for this.",
        "{verb} now or face {noun}.",
        "I'll make sure {noun} knows about this.",
        "You're going to {verb} for this.",
        "I'll take {noun} if you don't {verb}.",
        "I'm {verb} you to the {noun}.",
        "You'll be {verb} from my {noun}.",
        "I'll {verb} you for this.",
        "{verb} or I'll {verb2} your {noun}.",
        "I'll {verb} {noun} if you don't stop.",
        "This will be {verb} to the {noun}.",
        "I'm {verb} a {noun} against you.",
        "You'll {verb} from my {noun}.",
        "{verb} immediately or I'll {verb2}.",
        "I'll make this {verb} for you.",
        "You're going to {verb} this {noun}.",
        "I'll {verb} everyone about this.",
        "This is going to the {noun}.",
        "{verb} or face {adjective} {noun}.",
        "I'll have you {verb} for this."
    ],
    
    "Emotional": [
        "This reminded me of my {noun}.",
        "This made me {verb}. So {adjective}.",
        "I felt so {adjective} to this.",
        "This brings back so many {noun}.",
        "I'm in {noun}. This is {adjective}.",
        "This {verb} my {noun} {adverb}.",
        "I felt every {noun} {verb} this.",
        "This {verb} with me so much.",
        "I'm {adjective} with {noun}.",
        "This hit me right in the {noun}.",
        "I can't stop {verb} about this.",
        "This {verb} me to {noun}.",
        "I felt so {adjective} {verb} this.",
        "This brought {noun} to my {noun2}.",
        "I'm so {verb} by this.",
        "This {verb} up so many {noun}.",
        "I felt a {adjective} {noun} to this.",
        "This made me {verb} on my {noun}.",
        "I'm feeling so {adjective} right now.",
        "This {verb} my {noun}.",
        "I'm {verb} {adjective} {noun} right now.",
        "This {verb} to my {noun}.",
        "I felt this in my {noun}.",
        "This is so {adjective} to me.",
        "I'm {adverb} {adjective} in this.",
        "This {verb} something in me.",
        "I felt every {noun} of this.",
        "This is {adverb} {adjective} to me.",
        "I'm having all the {noun} right now.",
        "This {verb} exactly how I {verb2}."
    ],
    
    "Irrelevant/Spam": [
        "{verb} me for {noun}!",
        "Check out my {noun}!",
        "Click here for {adjective} {noun}!",
        "{verb} to me!",
        "{verb} this {noun} now!",
        "Visit my {noun} for more!",
        "{verb} me for {noun}!",
        "Get {adjective} {adverb}! Click here!",
        "{adjective} {noun}! {verb} me!",
        "{verb} and {verb2}!",
        "Check my {noun} for {noun2}!",
        "{verb} for {verb2} back!",
        "{verb} your {noun} here!",
        "{verb} a {noun}! Click now!",
        "{adjective} {noun}! Act {adverb}!",
        "Join my {noun} for {noun2}!",
        "Get {noun} {adverb}!",
        "Click the {noun} in my {noun2}!",
        "{adjective} {noun} just for you!",
        "{verb} me for {adjective} {noun}!",
        "Make {noun} from {noun2}! Click here!",
        "{verb} for a {noun}!",
        "{adjective} {noun} {noun2}! Enter now!",
        "{verb} {noun} {adverb}! Link in {noun2}!",
        "{verb} back everyone who {verb2} me!",
        "{verb} now and get {adjective} {noun}!",
        "{verb} to my {noun} for {noun2}!",
        "{adjective} {noun} available! {verb} now!",
        "{verb} your {noun} instantly!",
        "Don't miss this {adjective} {noun}!"
    ],
    
    "Question/Suggestion": [
        "Can you {verb} one on {noun}?",
        "What {noun} did you use for this?",
        "How {adjective} did this take to {verb}?",
        "Could you do a {noun} on this?",
        "What's your {noun} for {verb} this?",
        "Can you {verb} the {noun} you used?",
        "Would you {verb} making a {noun}?",
        "How did you {verb} to do this?",
        "What {verb} you to {verb2} this?",
        "Can you {verb} how you did this {noun}?",
        "Would love to see more {adjective} this!",
        "Could you make a {noun} video?",
        "What {noun} do you {verb} for {noun2}?",
        "How can I get {verb} with this?",
        "Can you {verb} your {noun}?",
        "Would you {verb} with {noun}?",
        "What's your next {noun}?",
        "Can you do a {noun} about this?",
        "How do you stay {adjective}?",
        "What's your {noun} for {adjective} {noun2}?",
        "Where did you {verb} these {noun}?",
        "Can you make a {noun} for {noun2}?",
        "What {noun} did you use?",
        "How much {noun} do you {verb} on each {noun2}?",
        "Could you {verb} this {noun} in more {noun2}?",
        "What's the best {noun} to {verb} this?",
        "Can you {verb} any {noun}?",
        "How do you {verb} your {noun}?",
        "What's your {noun} {noun2}?",
        "Would you {verb} doing {adjective} {noun}?"
    ]
}

# Word banks for template filling
WORD_BANKS = {
    "adjective": ["amazing", "incredible", "fantastic", "brilliant", "awesome", "excellent", "outstanding", 
                  "superb", "wonderful", "perfect", "beautiful", "stunning", "phenomenal", "magnificent",
                  "terrible", "awful", "horrible", "pathetic", "disgusting", "trash", "garbage", "worthless",
                  "good", "nice", "decent", "okay", "solid", "great", "fine", "impressive", "strong",
                  "poor", "weak", "bad", "rough", "choppy", "loud", "bright", "low", "slow", "long",
                  "touching", "emotional", "beautiful", "meaningful", "powerful", "deep", "moving",
                  "free", "limited", "special", "exclusive", "instant", "daily", "best", "top"],
    
    "adjective2": ["inspiring", "impressive", "creative", "talented", "skilled", "professional", "polished",
                   "useless", "incompetent", "shameful", "embarrassing", "disappointing", "unacceptable",
                   "better", "smoother", "clearer", "sharper", "faster", "shorter", "engaging", "natural",
                   "nostalgic", "personal", "profound", "resonant", "heartfelt", "genuine",
                   "time", "offer", "discount", "deal", "chance", "opportunity"],
    
    "noun": ["work", "content", "animation", "video", "effort", "job", "quality", "thing", "creation",
             "talent", "skill", "ability", "hope", "momentum", "progress", "dedication", "journey",
             "voiceover", "pacing", "execution", "audio", "story", "polish", "ending", "colors",
             "presentation", "transitions", "music", "timing", "editing", "text", "flow", "resolution",
             "trash", "garbage", "crap", "waste", "joke", "failure", "skill", "platform",
             "action", "consequences", "authorities", "lawyer", "complaint", "reputation",
             "childhood", "tears", "memories", "feelings", "heart", "soul", "core", "life",
             "followers", "channel", "stuff", "product", "website", "collaboration", "giveaway",
             "topic", "software", "tutorial", "process", "resources", "tools", "workflow", "advice"],
    
    "noun2": ["recognition", "support", "this", "work", "improvement", "detail", "refinement", "clarity",
              "time", "wasting", "ruining", "hearing", "filing", "exposing",
              "eyes", "feels", "thinking", "reflects", "captures",
              "updates", "content", "deals", "tips", "money", "home", "bio",
              "beginners", "depth", "courses", "creators", "aspiring"],
    
    "verb": ["loved", "enjoyed", "appreciated", "liked", "watched", "seen", "created", "made",
             "quit", "stop", "delete", "give up", "embarrassing", "wasting", "ruining",
             "improving", "working", "pushing", "believing", "rooting", "support",
             "improve", "be better", "read", "see", "understand", "follow",
             "report", "take", "regret", "stop", "face", "pay", "expose", "ruin", "file",
             "cry", "felt", "touched", "moved", "resonated", "awakened", "speaks", "captures",
             "follow", "check", "click", "subscribe", "buy", "visit", "get", "join", "promote", "win",
             "make", "use", "learn", "create", "share", "consider", "explain", "recommend", "plan", "stay"],
    
    "verb2": ["continues", "stops", "happens", "going", "at it", "forward", "in yourself",
              "better", "smoother", "clearer", "engaging",
              "banned", "reported", "exposed", "ruined",
              "feel", "think", "want", "need",
              "subscribe", "follow", "like", "promote", "enter",
              "inspired", "motivated", "started", "tutorials", "paid"],
    
    "aspect": ["animation", "voiceover", "pacing", "concept", "audio", "visuals", "idea", "ending",
               "colors", "content", "transitions", "music", "timing", "text", "flow", "resolution",
               "dialogue", "lighting", "graphics", "structure", "intro", "narrative", "sound",
               "message", "delivery", "foundation", "direction", "organization", "approach"],
    
    "aspect2": ["voiceover", "pacing", "execution", "quality", "story", "ending", "presentation",
                "music", "editing", "flow", "dialogue", "pace", "content", "mixing", "graphics",
                "structure", "organization", "intro", "narrative", "technical"],
    
    "time": ["today", "all day", "this week", "this month", "recently", "ever", "in a while",
             "this year", "lately", "so far", "week", "day", "month", "year"],
    
    "adverb": ["really", "so", "very", "extremely", "absolutely", "totally", "completely",
               "deeply", "truly", "immediately", "quickly", "fast", "now", "instantly",
               "emotionally", "personally", "profoundly", "right now", "exactly"]
}

def generate_comment(category):
    """Generate a single comment for the given category"""
    template = random.choice(TEMPLATES[category])
    
    # Fill in template placeholders
    comment = template
    for placeholder, words in WORD_BANKS.items():
        if "{" + placeholder + "}" in comment:
            comment = comment.replace("{" + placeholder + "}", random.choice(words))
    
    return comment

def generate_dataset(total_comments=100200):
    """Generate a large dataset of labeled comments"""
    
    # Distribution across categories (roughly balanced)
    category_distribution = {
        "Praise": 16000,
        "Support": 16000,
        "Constructive Criticism": 16000,
        "Hate/Abuse": 12000,
        "Threat": 10000,
        "Emotional": 15000,
        "Irrelevant/Spam": 15000,
        "Question/Suggestion": 16200  # Total = 100,200
    }
    
    print("Generating 100,200 labeled comments...")
    print("=" * 60)
    
    comments = []
    
    for category, count in category_distribution.items():
        print(f"Generating {count:,} {category} comments...")
        for i in range(count):
            comment = generate_comment(category)
            comments.append({
                'comment': comment,
                'category': category
            })
            
            # Progress indicator
            if (i + 1) % 5000 == 0:
                print(f"  Progress: {i + 1:,}/{count:,}")
    
    # Shuffle the dataset
    print("\nShuffling dataset...")
    random.shuffle(comments)
    
    # Save to CSV
    output_file = 'data/comments_dataset.csv'
    print(f"\nSaving to {output_file}...")
    
    df = pd.DataFrame(comments)
    df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL)
    
    print("\n" + "=" * 60)
    print("✅ Dataset generation complete!")
    print(f"Total comments: {len(df):,}")
    print("\nCategory distribution:")
    print(df['category'].value_counts().sort_index())
    print(f"\nDataset saved to: {output_file}")
    print(f"File size: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    return df

if __name__ == "__main__":
    df = generate_dataset(100200)
