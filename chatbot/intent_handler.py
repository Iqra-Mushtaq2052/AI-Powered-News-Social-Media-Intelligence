import re

class IntentHandler:
    """Classifies user queries into intelligence chatbot intents across all project topics."""
    
    INTENT_PATTERNS = {
        "GREETING": r"\b(hi|hello|hey|namaste|greetings|start|help|kaise ho|who are you)\b",
        "CONCEPT_EXPLANATION": r"\b(what is|explain|define|definition|how does|meaning|kaise kaam|kia hai|kia hota)\b",
        "SENTIMENT_SUMMARY": r"\b(sentiment summary|overall mood|overall sentiment|sentiment percentage|polarity score|vader summary|sentiment breakdown)\b",
        "MODEL_COMPARISON": r"\b(compare models|model benchmark|model comparison|compare ml|accuracy vs f1|compare lstm|benchmark metrics)\b",
        "NEWS_SUMMARY": r"\b(top headlines|latest news|headline summary|recent articles|recent updates|taza khabrein)\b",
        "SYSTEM_ARCHITECTURE": r"\b(architecture|pipeline|how system works|scraping pipeline|nlp pipeline|code breakdown|system flow)\b",
        "VIVA_EXPLANATION": r"\b(project purpose|maqsad|use case|why this project|viva|presentation|benefits)\b",
        "SOURCE_FILTER": r"\b(bbc|techcrunch|nyt|new york times|google news|reddit|r/technology|r/artificial|r/news)\b",
        "TOPIC_SEARCH": r"\b(ai|tech|gpu|semiconductor|chip|energy|startup|regulation|ethic|breaking|apple|google|microsoft|amazon|funding|market|economy|business)\b"
    }

    @classmethod
    def detect_intent(cls, user_query: str) -> str:
        """Detect the intent of the user query."""
        query_lower = user_query.lower()

        for intent, pattern in cls.INTENT_PATTERNS.items():
            if re.search(pattern, query_lower):
                return intent
                
        return "CONCEPT_EXPLANATION"
