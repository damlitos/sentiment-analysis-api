import uvicorn
from fastapi import FastAPI, Query, APIRouter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pydantic import BaseModel

app = FastAPI(title="Sentiment Model API",
    description="A simple API that use Vader to predict the sentiment",
    version="0.1")


analyzer = SentimentIntensityAnalyzer()

# Input and output models using Pydantic
class SentimentAnalysisRequest(BaseModel):
    text: str

class SentimentAnalysisResponse(BaseModel):
    text: str
    sentiment: str


router = APIRouter()

@router.post("/analyze_sentiment/", response_model=SentimentAnalysisResponse)
async def analyze_sentiment(request: SentimentAnalysisRequest):

    sentiment_scores = analyzer.polarity_scores(request.text)
    compound_score = sentiment_scores['compound']

    if compound_score >= 0.05:
        sentiment = "positive"
    elif compound_score <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return {"text": request.text, "sentiment": sentiment}

# multiple texts in a single request
@router.post("/analyze_multiple_sentiments/", response_model=list[SentimentAnalysisResponse])
async def analyze_multiple_sentiments(requests: list[SentimentAnalysisRequest]):
    results = []
    for req in requests:
        sentiment_scores = analyzer.polarity_scores(req.text)
        compound_score = sentiment_scores['compound']
        if compound_score >= 0.05:
            sentiment = "positive"
        elif compound_score <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        results.append({"text": req.text, "sentiment": sentiment})
    return results


@router.post("/analyze_sentiment_with_language/", response_model=SentimentAnalysisResponse)
async def analyze_sentiment_with_language(request: SentimentAnalysisRequest, language: str = Query("en", description="Language code (e.g., 'en' for English)")):

    return {"text": request.text, "sentiment": "positive", "language": language}


app.include_router(router)

if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)
