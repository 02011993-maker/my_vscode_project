from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib

# Load model
model = joblib.load("sentiment_model.pkl")

# Reverse label map
label_map = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

# FastAPI app
app = FastAPI(title="ML Tweet Sentiment")

# Input schema
class Tweet(BaseModel):
    text: str

@app.post("/predict")
async def predict(tweet: Tweet):
    pred = model.predict([tweet.text])[0]
    proba = model.predict_proba([tweet.text])[0][pred]
    return {
        "text": tweet.text,
        "label": label_map[pred],
        "score": round(float(proba), 4)
    }

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
   <html>
    <head>
        <title>Tweet Sentiment Predictor</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: url('https://www.toptal.com/designers/subtlepatterns/uploads/doodles.png');
                background-size: repeat;
                margin: 0;
                padding: 0;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                min-height: 100vh;
                background-color: #f5f8ff;
            }
            .container {
                background: #ffffffd9;
                padding: 40px;
                border-radius: 16px;
                box-shadow: 0 8px 24px rgba(0,0,0,0.15);
                text-align: center;
                max-width: 600px;
                width: 90%;
            }
            h2 {
                color: #003e66;
            }
            input[type='text'] {
                width: 80%;
                padding: 12px;
                margin: 10px 0;
                border: 2px solid #ccc;
                border-radius: 8px;
                font-size: 16px;
            }
            button {
                padding: 12px 24px;
                margin: 10px 5px;
                font-size: 16px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                transition: 0.3s;
            }
            #analyzeBtn {
                background-color: #ffb400;
                color: #003e66;
            }
            #saveBtn {
                background-color: #003e66;
                color: white;
            }
            #analyzeBtn:hover {
                background-color: #e0a000;
            }
            #saveBtn:hover {
                background-color: #002b4d;
            }
            #result {
                margin-top: 20px;
                font-size: 18px;
                color: #222;
                background-color: #eef3ff;
                padding: 16px;
                border-radius: 10px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🎯 Tweet Sentiment Predictor</h2>
            <form id="tweetForm">
                <input type="text" id="tweetText" placeholder="Type your tweet here..." required />
                <br />
                <button type="submit" id="analyzeBtn">Analyze</button>
                <button type="button" id="saveBtn">Save Prediction</button>
            </form>
            <div id="result"></div>
        </div>

        <script>
            let currentPrediction = null;

            document.getElementById("tweetForm").addEventListener("submit", async function(e) {
                e.preventDefault();
                const text = document.getElementById("tweetText").value;

                const response = await fetch("/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ text })
                });

                const result = await response.json();
                currentPrediction = result;

                document.getElementById("result").innerHTML = `
                    <p><strong>Sentiment:</strong> ${result.label}</p>
                    <p><strong>Confidence:</strong> ${result.score}</p>
                `;
            });

            document.getElementById("saveBtn").addEventListener("click", function() {
                if (currentPrediction) {
                    alert("Prediction saved! (This is just a UI demo, add backend save logic)");
                    // TODO: send POST to /save if needed
                } else {
                    alert("Please analyze a tweet first.");
                }
            });
        </script>
    </body>
    </html>
    """
