"""
Response Templates for Different Comment Categories
This module provides suggested response templates for each category.
"""

RESPONSE_TEMPLATES = {
    "Praise": [
        "Thank you so much for your kind words! We're thrilled you enjoyed it! 🙏",
        "We really appreciate your support! Comments like yours keep us motivated! ❤️",
        "So glad you loved it! Stay tuned for more content coming soon! ✨"
    ],
    
    "Support": [
        "Thank you for the encouragement! Your support means the world to us! 💪",
        "We appreciate you believing in us! We'll keep working hard! 🙌",
        "Your words motivate us to keep pushing forward! Thank you! 🌟"
    ],
    
    "Constructive Criticism": [
        "Thank you for the valuable feedback! We'll definitely work on improving that aspect. 🙏",
        "We appreciate your honest input! This helps us grow and improve our content. 📈",
        "Great point! We'll take this into consideration for our next project. Thanks! 💡"
    ],
    
    "Hate/Abuse": [
        "[Escalate to moderation team - Do not engage directly]",
        "[Flag for review - Consider blocking if harassment continues]",
        "[Report and ignore - Do not respond publicly]"
    ],
    
    "Threat": [
        "[URGENT: Escalate to legal/security team immediately]",
        "[Document and report to platform administrators]",
        "[Do not engage - Forward to appropriate authorities]"
    ],
    
    "Emotional": [
        "We're so touched that this resonated with you! Thank you for sharing. ❤️",
        "Your emotional connection means everything to us. Thank you! 🙏",
        "We're glad this touched your heart. Thank you for being part of our community! 💙"
    ],
    
    "Irrelevant/Spam": [
        "[Ignore or mark as spam - Do not engage]",
        "[Report as spam and hide comment]",
        "[Block user if spam continues]"
    ],
    
    "Question/Suggestion": [
        "Great question! We'll consider creating content on this topic. Thanks for the suggestion! 💡",
        "Thanks for asking! [Provide specific answer or direct to resources] 📚",
        "We love this idea! We'll add it to our content pipeline. Stay tuned! 🎬"
    ]
}


def get_response_template(category, index=0):
    """
    Get a response template for a specific category.
    
    Args:
        category (str): The comment category
        index (int): Index of the template to retrieve (default: 0)
        
    Returns:
        str: Response template
    """
    templates = RESPONSE_TEMPLATES.get(category, ["No template available for this category."])
    return templates[index % len(templates)]


def get_all_templates(category):
    """
    Get all response templates for a specific category.
    
    Args:
        category (str): The comment category
        
    Returns:
        list: List of all response templates for the category
    """
    return RESPONSE_TEMPLATES.get(category, ["No template available for this category."])


def get_action_recommendation(category):
    """
    Get recommended action for each category.
    
    Args:
        category (str): The comment category
        
    Returns:
        str: Recommended action
    """
    actions = {
        "Praise": "✅ Engage positively - Respond with gratitude",
        "Support": "✅ Engage positively - Show appreciation",
        "Constructive Criticism": "✅ Address feedback - Show willingness to improve",
        "Hate/Abuse": "⚠️ Escalate - Do not engage, report to moderation",
        "Threat": "🚨 URGENT - Escalate to legal/security immediately",
        "Emotional": "✅ Engage empathetically - Acknowledge their feelings",
        "Irrelevant/Spam": "🗑️ Ignore/Delete - Mark as spam",
        "Question/Suggestion": "✅ Provide answer - Engage constructively"
    }
    return actions.get(category, "No recommendation available")
