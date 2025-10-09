// ----------------------------
// Toggle between Text and URL input
// ----------------------------
const toggleSwitch = document.getElementById("toggleSwitch");
const textWrapper = document.getElementById("textInputWrapper");
const urlWrapper = document.getElementById("urlInputWrapper");

let usingText = true;

// Initialize default mode (Text Analysis)
toggleSwitch.classList.remove("active"); // No 'active' = Text mode
textWrapper.style.display = "block";
urlWrapper.style.display = "none";

toggleSwitch.addEventListener("click", () => {
  usingText = !usingText;
  toggleSwitch.classList.toggle("active", !usingText); // 'active' = URL mode

  if (usingText) {
    textWrapper.style.display = "block";
    urlWrapper.style.display = "none";
    console.log("Switched to TEXT input");
  } else {
    textWrapper.style.display = "none";
    urlWrapper.style.display = "block";
    console.log("Switched to URL input");
  }
});

// ----------------------------
// Mock Analysis Results
// (Replace this section with your Flask API call later)
// ----------------------------
// Show the Analysis Summary section dynamically

function showAnalysisSummary() {
  const summarySection = document.getElementById("analysis-summary");
  summarySection.style.display = "block";
}

// Example: Button click handler for your "Analyze" button
document.getElementById("analyzeBtn").addEventListener("click", function() {
  const analyzeBtn = this;
  const resultsDiv = document.getElementById("results");

  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing...";

  // Mock delay for analysis
  setTimeout(() => {
    // Example mock results
    const mockResults = {
      summary: "Content analyzed successfully",
      biasScore: 45,
      fakeNewsScore: 18,
    };

    // Update your results area
    document.getElementById("summary").textContent = mockResults.summary;
    document.getElementById("biasScore").textContent = `Bias: ${mockResults.biasScore}%`;
    document.getElementById("fakeNewsScore").textContent = `Fake News Risk: ${mockResults.fakeNewsScore}%`;
    resultsDiv.style.display = "block";

    // Show Analysis Summary
    showAnalysisSummary();

    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyze Content";
  }, 2000);
});


analyzeBtn.addEventListener("click", async () => {
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing...";

  // Simulate delay for analysis
  setTimeout(() => {
    document.getElementById("summary").textContent = mockResults.summary;
    document.getElementById("biasScore").textContent = `Bias: ${mockResults.biasScore}%`;
    document.getElementById("fakeNewsScore").textContent = `Fake News Risk: ${mockResults.fakeNewsScore}%`;

    resultsDiv.style.display = "block";
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyze Content";
  }, 2000);
});

// ----------------------------
// Feedback System
// ----------------------------
const stars = document.querySelectorAll("#stars span");
let rating = 0;

stars.forEach(star => {
  star.addEventListener("click", () => {
    rating = star.dataset.val;
    stars.forEach(s => {
      s.classList.toggle("active", s.dataset.val <= rating);
    });
  });
});

document.getElementById("feedbackBtn").addEventListener("click", () => {
  if (rating > 0) {
    document.getElementById("feedbackMsg").style.display = "block";
  } else {
    alert("Please select a rating before submitting feedback.");
  }
});
