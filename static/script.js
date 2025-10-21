// ========================================================
// TOGGLE BETWEEN TEXT AND URL INPUT
// ========================================================
const toggleSwitch = document.getElementById("toggleSwitch");
const textWrapper = document.getElementById("textInputWrapper");
const urlWrapper = document.getElementById("urlInputWrapper");

let usingText = true;

// Default mode: Text Analysis
toggleSwitch.classList.remove("active");
textWrapper.style.display = "block";
urlWrapper.style.display = "none";

toggleSwitch.addEventListener("click", () => {
  usingText = !usingText;
  toggleSwitch.classList.toggle("active", !usingText);

  if (usingText) {
    textWrapper.style.display = "block";
    urlWrapper.style.display = "none";
  } else {
    textWrapper.style.display = "none";
    urlWrapper.style.display = "block";
  }
});

// ========================================================
// ANALYZE CONTENT BUTTON
// ========================================================
document.getElementById("analyzeBtn").addEventListener("click", async function () {
  const analyzeBtn = this;
  const resultsDiv = document.getElementById("results");

  let userInput, inputType;
  if (usingText) {
    userInput = document.getElementById("inputText").value;
    inputType = "text";
  } else {
    userInput = document.getElementById("inputUrl").value;
    inputType = "url";
  }

  if (!userInput.trim()) {
    alert("Please enter some text or a URL to analyze.");
    return;
  }

  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing...";

  try {
    const formData = new FormData();
    formData.append("text", userInput);
    formData.append("input_type", inputType);

    const response = await fetch("/analyze", {
      method: "POST",
      headers: { "Accept": "application/json" },
      body: formData,
    });

    if (response.ok) {
      const result = await response.json();
      displayResults(result);
      resultsDiv.style.display = "block";
      resultsDiv.scrollIntoView({ behavior: "smooth" });
    } else {
      throw new Error("Analysis failed");
    }
  } catch (error) {
    console.error("Error:", error);
    displayMockResults(); // fallback for demo
    resultsDiv.style.display = "block";
    resultsDiv.scrollIntoView({ behavior: "smooth" });
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyze Content";
  }
});

// ========================================================
// DISPLAY RESULTS FUNCTION
// ========================================================
function displayResults(result) {
  // ----- Basic Overview -----
  document.getElementById("wordsAnalyzed").textContent = result.words_analyzed;
  document.getElementById("biasScore").textContent = `Bias: ${result.bias_score}%`;
  document.getElementById("fakeNewsScore").textContent = `Fake News Risk: ${result.fake_news_risk}%`;

  // ----- Domain Data -----
  document.getElementById("reliableScore").textContent = `${result.domain_data_score}% Reliable devices`;
  document.getElementById("redFlagScore").textContent = `${result.user_computer_data}% Red flags`;

  // ----- Language and Tone -----
  document.getElementById("emotionalWords").textContent = `${result.emotional_words_percentage}%`;
  document.getElementById("sourceReliability").textContent = result.source_reliability;
  document.getElementById("framingPerspective").textContent = result.framing_perspective;

  // ----- Sentiment -----
  document.getElementById("positiveSentiment").textContent = `${result.positive_sentiment}%`;
  document.getElementById("negativeSentiment").textContent = `${result.negative_sentiment}%`;

  // ----- Word Repetition -----
  const wordList = document.getElementById("wordRepetitionList");
  wordList.innerHTML = "";
  result.word_repetition.forEach((wordData) => {
    const li = document.createElement("li");
    li.innerHTML = `<strong>${wordData.word}:</strong> ${wordData.count} occurrences`;
    wordList.appendChild(li);
  });

  // ----- Summary Sections -----
  document.getElementById("overviewText").textContent =
    result.overview || "This content demonstrates moderate political bias but maintains factual accuracy.";
  document.getElementById("reliabilityText").textContent =
    result.reliability || "Most sources appear trustworthy, with minor subjective language detected.";
  document.getElementById("recommendationText").textContent =
    result.recommendation || "Cross-check similar sources to confirm facts and reduce potential framing bias.";

  // ----- Key Points -----
  const keyPointsList = document.getElementById("keyPointsList");
  keyPointsList.innerHTML = `
    <li>Emotionally charged words: <b>${result.emotional_words_percentage}%</b></li>
    <li>Source reliability: <b>${result.source_reliability}</b></li>
    <li>Overall tone: <b>${result.overall_tone}</b></li>
  `;

  // ----- Sentiment Chart -----
  if (window.sentimentChart instanceof Chart) {
    window.sentimentChart.destroy();
  }
  const ctxSentiment = document.getElementById("sentimentChart");
  window.sentimentChart = new Chart(ctxSentiment, {
    type: "bar",
    data: {
      labels: ["Positive", "Negative"],
      datasets: [
        {
          label: "Sentiment Distribution",
          data: [result.positive_sentiment, result.negative_sentiment],
          backgroundColor: ["#4CAF50", "#F44336"],
        },
      ],
    },
    options: { responsive: true },
  });

  // ----- Bias Detection -----
  if (result.bias_analysis) {
    const bias = result.bias_analysis;
    document.getElementById("biasPrediction").textContent = bias.bias_prediction;
    document.getElementById("biasConfidence").textContent = `${(bias.confidence * 100).toFixed(1)}%`;

    // Bias Chart
    if (window.biasChart instanceof Chart) {
      window.biasChart.destroy();
    }
    const ctxBias = document.getElementById("biasChart").getContext("2d");
    window.biasChart = new Chart(ctxBias, {
      type: "bar",
      data: {
        labels: ["Left", "Center", "Right"],
        datasets: [{
          label: "Bias Confidence (%)",
          data: [
            bias.bias_prediction === "left" ? bias.confidence * 100 : 0,
            bias.bias_prediction === "center" ? bias.confidence * 100 : 0,
            bias.bias_prediction === "right" ? bias.confidence * 100 : 0
          ],
          backgroundColor: ["#FF6384", "#36A2EB", "#FFCE56"]
        }]
      },
      options: {
        responsive: true,
        scales: { y: { beginAtZero: true, max: 100 } }
      }
    });
  }


  // ----- Word Frequency Chart -----
  if (window.wordChart instanceof Chart) {
    window.wordChart.destroy();
  }
  const ctxWord = document.getElementById("wordChart");
  const labels = result.word_repetition.map((w) => w.word);
  const counts = result.word_repetition.map((w) => w.count);
  window.wordChart = new Chart(ctxWord, {
    type: "pie",
    data: {
      labels: labels,
      datasets: [
        {
          data: counts,
          backgroundColor: ["#FF6384", "#36A2EB", "#FFCE56", "#9CCC65", "#FF7043"],
        },
      ],
    },
    options: { responsive: true },
  });
}

// ========================================================
// MOCK RESULTS FOR FALLBACK
// ========================================================
function displayMockResults() {
  const mockResult = {
    words_analyzed: 450,
    domain_data_score: 72,
    user_computer_data: 28,
    emotional_words_percentage: 12,
    source_reliability: "High",
    positive_sentiment: 90,
    negative_sentiment: 84,
    word_repetition: [
      { word: "Crisis", count: 22 },
      { word: "Government warned", count: 18 },
      { word: "Urgent", count: 16 },
    ],
    framing_perspective: "Recent reading via government sources.",
    overall_tone: "Balanced but critical",
    bias_score: 45,
    fake_news_risk: 18,
    overview: "This content demonstrates moderate political bias but maintains factual accuracy.",
    reliability: "Most sources appear trustworthy, with minor subjective language detected.",
    recommendation: "Cross-check similar sources to confirm facts and reduce potential framing bias.",
  };

  displayResults(mockResult);
}

// FEEDBACK SYSTEM
const stars = document.querySelectorAll("#stars span");
let rating = 0;

stars.forEach((star) => {
  star.addEventListener("click", () => {
    rating = star.dataset.val;
    stars.forEach((s) => {
      s.classList.toggle("active", s.dataset.val <= rating);
    });
  });
});

document.getElementById("feedbackBtn").addEventListener("click", () => {
  if (rating > 0) {
    document.getElementById("feedbackMsg").style.display = "block";
    setTimeout(() => {
      document.getElementById("feedbackMsg").style.display = "none";
    }, 3000);
  } else {
    alert("Please select a rating before submitting feedback.");
  }
});
