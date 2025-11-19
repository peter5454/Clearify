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
  // ----- Bias -----
  let biasScoreNum = 0;
  if (Array.isArray(result.bias_score)) {
    biasScoreNum = result.bias_score[0];
  } else if (typeof result.bias_score === "number") {
    biasScoreNum = result.bias_score;
  } else if (result.bias_score?.score) {
    biasScoreNum = result.bias_score.score;
  }

  const isBiased = biasScoreNum >= 50; // Threshold for bias
  const biasLabelText = isBiased ? "Biased" : "Non Biased";

  const biasElement = document.getElementById("biasLabel");
  biasElement.textContent = `Bias: ${biasLabelText} (${Math.round(biasScoreNum)}%)`;
  biasElement.style.color = isBiased ? "red" : "green";
  biasElement.style.fontWeight = "600";

  // ----- Words Analyzed -----
  document.getElementById("wordsAnalyzed").textContent = result.words_analyzed ?? 0;

  // ----- Fake News -----
  document.getElementById("fakeNewsScore").textContent = `Fake News Risk: ${result.fake_news_risk ?? 0}%`;

  // ----- Emotional Words -----
  document.getElementById("emotionalWords").textContent = `${result.emotional_words_percentage ?? 0}%`;

  // ----- Sentiment -----
  document.getElementById("positiveSentiment").textContent = `${result.positive_sentiment ?? 0}%`;
  document.getElementById("negativeSentiment").textContent = `${result.negative_sentiment ?? 0}%`;

  // ----- Word Repetition -----
  const wordList = document.getElementById("wordRepetitionList");
  wordList.innerHTML = "";
  (result.word_repetition || []).forEach((wordData) => {
    const li = document.createElement("li");
    li.innerHTML = `<strong>${wordData.word}:</strong> ${wordData.count} occurrence${wordData.count === 1 ? "" : "s"}`;
    wordList.appendChild(li);
  });

  // ----- Sentiment Chart -----
  if (window.sentimentChart instanceof Chart) window.sentimentChart.destroy();
  const ctxSentiment = document.getElementById("sentimentChart");
  window.sentimentChart = new Chart(ctxSentiment, {
    type: "bar",
    data: {
      labels: ["Positive", "Negative"],
      datasets: [{
        label: "Sentiment",
        data: [result.positive_sentiment ?? 0, result.negative_sentiment ?? 0],
        backgroundColor: ["#4CAF50", "#F44336"]
      }],
    },
    options: { responsive: true },
  });

  // ----- Word Frequency Chart -----
  if (window.wordChart instanceof Chart) window.wordChart.destroy();
  const ctxWord = document.getElementById("wordChart");
  const labels = (result.word_repetition || []).map((w) => w.word);
  const counts = (result.word_repetition || []).map((w) => w.count);
  window.wordChart = new Chart(ctxWord, {
    type: "pie",
    data: { labels, datasets: [{ data: counts, backgroundColor: ["#FF6384", "#36A2EB", "#FFCE56", "#9CCC65", "#FF7043"] }] },
    options: { responsive: true },
  });

  // ----- Political Analysis -----
  const pol = result.political_analysis || {};
  document.getElementById("politicalLabel").textContent = (pol.prediction || "UNKNOWN").toUpperCase();
  const polConf = Math.round((pol.confidence || 0) * 100);
  document.getElementById("politicalConfidence").textContent = `${polConf}%`;
  document.getElementById("politicalProgress").style.width = `${polConf}%`;

  // ----- Social Bias Analysis -----
  const sb = result.social_bias_analysis || {};
  document.getElementById("socialLabel").textContent = (sb.bias_category || "None").replace(/^\w/, (c) => c.toUpperCase());
  const sbConf = Math.round((sb.confidence || 0) * 100);
  document.getElementById("socialConfidence").textContent = `${sbConf}%`;
  document.getElementById("socialProgress").style.width = `${sbConf}%`;

  // ----- Gemini Summary -----
  const geminiCard = document.getElementById("gemini-summary-card");
  if (result.gemini_summary) {
    geminiCard.style.display = "block";
    document.getElementById("geminiOverall").textContent = result.gemini_summary.overall_summary || "—";
    document.getElementById("geminiPolitical").textContent = result.gemini_summary.political_bias_summary || "—";
    document.getElementById("geminiSocial").textContent = result.gemini_summary.social_bias_summary || "—";
    document.getElementById("geminiFakeNews").textContent = result.gemini_summary.fake_news_summary || "—";
    document.getElementById("geminiVerdict").textContent = result.gemini_summary.final_verdict || "—";
  } else {
    geminiCard.style.display = "none";
  }

  document.getElementById("results").style.display = "block";
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
    overview: "This content demonstrates moderate political bias but maintains factual accuracy.",
    reliability: "Most sources appear trustworthy, with minor subjective language detected.",
    recommendation: "Cross-check similar sources to confirm facts and reduce potential framing bias.",

  };

  displayResults(mockResult);
}
//FEEDBACK / STAR RATING SYSTEM
document.addEventListener("DOMContentLoaded", () => {
  const stars = document.querySelectorAll("#stars span");
  let rating = 0;

  stars.forEach((star) => {
    star.addEventListener("click", () => {
      rating = parseInt(star.dataset.val);
      stars.forEach((s) => {
        s.classList.toggle("active", parseInt(s.dataset.val) <= rating);
      });
    });
  });

  // ===== FEEDBACK SUBMISSION =====
  document.getElementById("feedbackBtn").addEventListener("click", async () => {
    const feedbackText = document.getElementById("feedbackText").value;
    const submittedText =
      document.getElementById("inputText").value ||
      document.getElementById("inputUrl").value;

    if (!rating) {
      alert("Please select a rating before submitting feedback.");
      return;
    }

    try {
      const response = await fetch("/submit_feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          rating,
          feedback_text: feedbackText,
          submitted_text: submittedText,
        }),
      });

      const result = await response.json();
      if (response.ok) {
        document.getElementById("feedbackMsg").style.display = "block";
        setTimeout(() => {
          document.getElementById("feedbackMsg").style.display = "none";
        }, 3000);
      } else {
        alert(result.error || "Failed to submit feedback.");
      }
    } catch (err) {
      console.error(err);
      alert("Failed to submit feedback.");
    }
  });
});




