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
  // ----- Support different bias_score shapes -----
  // backend may return bias_score as number or as [score,label]
  let biasScoreNum = 0;
  let dbiasLabel = "unknown";
  if (Array.isArray(result.bias_score)) {
    biasScoreNum = result.bias_score[0];
    dbiasLabel = result.bias_score[1] || "unknown";
  } else if (typeof result.bias_score === "number") {
    biasScoreNum = result.bias_score;
  } else {
    // If nested like {score:..., label:...}
    if (result.bias_score && typeof result.bias_score === "object") {
      biasScoreNum = result.bias_score.score || 0;
      dbiasLabel = result.bias_score.label || "unknown";
    }
  }

  // ----- Basic Overview -----
  document.getElementById("wordsAnalyzed").textContent = result.words_analyzed ?? 0;
  document.getElementById("biasScore").textContent = `Bias: ${Math.round(biasScoreNum)}%`;
  // show dbias label appended as small text (optional)
  // If you want to show label near the bias score:
  // document.getElementById("biasScore").textContent += ` (${dbiasLabel})`;

  document.getElementById("fakeNewsScore").textContent = `Fake News Risk: ${result.fake_news_risk ?? 0}%`;

  // ----- Domain Data -----
  document.getElementById("reliableScore").textContent = `${result.domain_data_score ?? 0}% Reliable devices`;
  document.getElementById("redFlagScore").textContent = `${result.user_computer_data ?? 0}% Red flags`;

  // ----- Language and Tone -----
  document.getElementById("emotionalWords").textContent = `${result.emotional_words_percentage ?? 0}%`;
  document.getElementById("sourceReliability").textContent = result.source_reliability ?? "Unknown";
  document.getElementById("framingPerspective").textContent = result.framing_perspective ?? "—";

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
    <li>Emotionally charged words: <b>${result.emotional_words_percentage ?? 0}%</b></li>
    <li>Source reliability: <b>${result.source_reliability ?? "Unknown"}</b></li>
    <li>Overall tone: <b>${result.overall_tone ?? "Unknown"}</b></li>
  `;

  // ----- Sentiment Chart (unchanged) -----
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
          data: [result.positive_sentiment ?? 0, result.negative_sentiment ?? 0],
          backgroundColor: ["#4CAF50", "#F44336"],
        },
      ],
    },
    options: { responsive: true },
  });

  // ----- Word Frequency Chart (unchanged) -----
  if (window.wordChart instanceof Chart) {
    window.wordChart.destroy();
  }
  const ctxWord = document.getElementById("wordChart");
  const labels = (result.word_repetition || []).map((w) => w.word);
  const counts = (result.word_repetition || []).map((w) => w.count);
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

  // ----- Political Analysis (new) -----
  // political_result expected: { prediction: "left"|"center"|"right", confidence: 0.### }
  const pol = result.political_analysis || {};
  const polPred = pol.prediction ? String(pol.prediction).toUpperCase() : "UNKNOWN";
  const polConf = typeof pol.confidence === "number" ? Math.round(pol.confidence * 100) : (pol.confidence || 0);

  document.getElementById("politicalLabel").textContent = polPred;
  document.getElementById("politicalConfidence").textContent = `${polConf}%`;
  const polProg = document.getElementById("politicalProgress");
  polProg.style.width = `${polConf}%`;
  // color tweak for common labels:
  polProg.classList.remove("right","center");
  if ((pol.prediction || "").toLowerCase() === "right") polProg.classList.add("right");
  if ((pol.prediction || "").toLowerCase() === "center") polProg.classList.add("center");

  // Optionally show predicted label near the top of summary or key points
  // add to keypoints
  keyPointsList.insertAdjacentHTML("beforeend", `<li>Political leaning: <b>${polPred} (${polConf}%)</b></li>`);

  // ----- Social Bias Analysis (new) -----
  // sbic_result expected: { bias_category: "race"|"gender"|..., confidence: 0.### }
  const sb = result.social_bias_analysis || {};
  const sbPred = sb.bias_category ? String(sb.bias_category) : "none";
  const sbConf = typeof sb.confidence === "number" ? Math.round(sb.confidence * 100) : (sb.confidence || 0);

  document.getElementById("socialLabel").textContent = sbPred.replace(/^\w/, (c) => c.toUpperCase());
  document.getElementById("socialConfidence").textContent = `${sbConf}%`;
  const sbProg = document.getElementById("socialProgress");
  sbProg.style.width = `${sbConf}%`;

  // add to keypoints
  keyPointsList.insertAdjacentHTML("beforeend", `<li>Social bias: <b>${sbPred} (${sbConf}%)</b></li>`);

  // Show results section if hidden
  const resultsDiv = document.getElementById("results");
  if (resultsDiv && resultsDiv.style.display === "none") {
    resultsDiv.style.display = "block";
  }
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
