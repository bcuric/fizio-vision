import {
  DrawingUtils,
  FilesetResolver,
  PoseLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21";
import {
  LANDMARKS,
  analyzeGait,
  analyzeKnee,
  legSample,
  median,
  medianSmooth,
  samplesCsv,
  toCsv,
} from "./analysis.js";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task";
const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm";
const MIN_VISIBILITY = 0.5;
const SMOOTHING_WINDOW = 5;
const COMMON_FRAME_RATES = [24, 25, 30000 / 1001, 30, 50, 60000 / 1001, 60];
const HISTORY_KEY = "fizio-vision-mjerenja-v1";
const AGREEMENT_LIMIT = 5;
// validirano dataviz/validate_palette.js na #0e1a17 (lijeva / desna noga)
const LEG_COLORS = { left: "#6ea83a", right: "#3f86d6", near: "#b7f36b" };

const video = document.querySelector("#camera");
const canvas = document.querySelector("#overlay");
const context = canvas.getContext("2d");
const viewer = document.querySelector("#viewer");
const viewerMessage = document.querySelector("#viewer-message");
const liveAngle = document.querySelector("#live-angle");
const appTitle = document.querySelector("#app-title");
const statusText = document.querySelector("#status");
const statusDot = document.querySelector("#status-dot");
const cameraButton = document.querySelector("#camera-button");
const recordButton = document.querySelector("#record-button");
const resetButton = document.querySelector("#reset-button");
const sideSelect = document.querySelector("#side-select");
const testMode = document.querySelector("#test-mode");
const sideControls = document.querySelector("#side-controls");
const testHint = document.querySelector("#test-hint");
const installButton = document.querySelector("#install-button");
const postureGuide = document.querySelector("#posture-guide");
const guideTitle = document.querySelector("#guide-title");
const guideStepCount = document.querySelector("#guide-step-count");
const guideDemo = document.querySelector("#guide-demo");
const guideDemoCaption = document.querySelector("#guide-demo-caption");
const guideCopy = document.querySelector("#guide-copy");
const guideProgress = document.querySelector("#guide-progress");
const guideDetectionDot = document.querySelector("#guide-detection-dot");
const resultsTitle = document.querySelector("#results-title");
const metricLabels = [1, 2, 3].map((number) =>
  document.querySelector(`#metric-label-${number}`),
);
const metricOutputs = [1, 2, 3].map((number) =>
  document.querySelector(`#metric-${number}`),
);
const resultNote = document.querySelector("#result-note");
const repetitionDetails = document.querySelector("#repetition-details");
const fileControls = document.querySelector("#file-controls");
const fileButton = document.querySelector("#file-button");
const fileInput = document.querySelector("#file-input");
const chart = document.querySelector("#chart");
const chartTitle = document.querySelector("#chart-title");
const chartCanvas = document.querySelector("#chart-canvas");
const chartTooltip = document.querySelector("#chart-tooltip");
const chartLegend = document.querySelector("#chart-legend");
const saveForm = document.querySelector("#save-form");
const patientInput = document.querySelector("#patient-input");
const goniometerField = document.querySelector("#goniometer-field");
const goniometerInput = document.querySelector("#goniometer-input");
const framesButton = document.querySelector("#frames-button");
const exportButton = document.querySelector("#export-button");
const clearHistoryButton = document.querySelector("#clear-history-button");
const historySummary = document.querySelector("#history-summary");
const historyTable = document.querySelector("#history-table");
const historyBody = document.querySelector("#history-body");

let landmarker;
let cameraStream;
let animationFrame;
let lastVideoTime = -1;
let lastDetectTimestamp = 0;
let isRecording = false;
let recordingStats = createPostureStats();
let legSamples = [];
let legSource = "kamera";
let currentMeasurement;
let chartData;
let fileAnalysis;
let recentFlexions = [];
let postureGuideStep = 0;
let posturePhase = "find-person";
let postureConditionSince = 0;
let postureStableMeasurements = [];
let postureResetBaseline;
let postureRepResults = [];
let recordingStartedAt = 0;
const POSTURE_REPETITIONS = 3;
const POSTURE_REPETITION_MS = 3000;
const SETUP_HOLD_MS = 900;
const FRONT_HOLD_MS = 900;
const NEUTRAL_HOLD_MS = 650;
let installPrompt;

function createPostureStats() {
  return { samples: [], totalFrames: 0 };
}

function isLegMode() {
  return testMode.value !== "posture";
}

/** MediaPipe u VIDEO načinu traži rastuće vremenske oznake, i kad se izmjenjuju kamera i snimka. */
function nextDetectTimestamp() {
  lastDetectTimestamp = Math.max(performance.now(), lastDetectTimestamp + 1);
  return lastDetectTimestamp;
}

/** Noga koja se prikazuje i mjeri u ovom frameu; u hodu i za „bliža” odlučuje z koljena. */
function frameSide(sample) {
  const side = testMode.value === "gait" ? "near" : sideSelect.value;
  if (side !== "near") return side;
  const nearLeft = sample.zL <= sample.zR;
  const near = nearLeft ? sample.angL : sample.angR;
  if (Number.isNaN(near)) return nearLeft ? "right" : "left";
  return nearLeft ? "left" : "right";
}

function setStatus(message, active = false) {
  statusText.textContent = message;
  statusDot.classList.toggle("is-active", active);
}

function setBusy(isBusy) {
  cameraButton.disabled = isBusy;
  cameraButton.textContent = isBusy ? "Učitavam model…" : "Pokreni kameru";
}

function resizeOverlay() {
  const { videoWidth, videoHeight } = video;
  if (!videoWidth || !videoHeight) return;
  if (canvas.width !== videoWidth || canvas.height !== videoHeight) {
    canvas.width = videoWidth;
    canvas.height = videoHeight;
  }
}

function drawPose(landmarks, side) {
  const drawing = new DrawingUtils(context);
  drawing.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
    color: "rgba(228, 242, 235, 0.48)",
    lineWidth: 3,
  });
  drawing.drawLandmarks(landmarks, {
    color: "rgba(228, 242, 235, 0.8)",
    lineWidth: 1,
    radius: 3,
  });

  const joints = LANDMARKS[side];
  const leg = [landmarks[joints.hip], landmarks[joints.knee], landmarks[joints.ankle]];
  drawing.drawConnectors(landmarks, [[joints.hip, joints.knee], [joints.knee, joints.ankle]], {
    color: "#b7f36b",
    lineWidth: 5,
  });
  drawing.drawLandmarks(leg, {
    color: "#b7f36b",
    lineWidth: 2,
    radius: 6,
  });
}

function postureMeasurement(landmarks) {
  const points = [landmarks[11], landmarks[12], landmarks[7], landmarks[8], landmarks[0]];
  if (points.some((point) => !point || (point.visibility ?? 0) < MIN_VISIBILITY)) {
    return null;
  }
  const [leftShoulder, rightShoulder, leftEar, rightEar, nose] = points.map((point) => ({
    x: point.x * video.videoWidth,
    y: point.y * video.videoHeight,
  }));
  const signedLineAngle = (a, b) => {
    let angle = (Math.atan2(b.y - a.y, b.x - a.x) * 180) / Math.PI;
    if (angle > 90) angle -= 180;
    if (angle < -90) angle += 180;
    return angle;
  };
  const earCenter = { x: (leftEar.x + rightEar.x) / 2, y: (leftEar.y + rightEar.y) / 2 };
  const shoulderCenter = {
    x: (leftShoulder.x + rightShoulder.x) / 2,
    y: (leftShoulder.y + rightShoulder.y) / 2,
  };
  const shoulderWidth = Math.hypot(
    rightShoulder.x - leftShoulder.x,
    rightShoulder.y - leftShoulder.y,
  );
  const earWidth = Math.hypot(rightEar.x - leftEar.x, rightEar.y - leftEar.y);
  if (shoulderWidth === 0 || earWidth === 0 || !video.videoWidth || !video.videoHeight) return null;
  const signedShoulderTilt = signedLineAngle(leftShoulder, rightShoulder);
  const signedHeadTilt = signedLineAngle(leftEar, rightEar);
  const shoulderCenterX = shoulderCenter.x / video.videoWidth;
  const noseCenterOffset = (nose.x - earCenter.x) / earWidth;
  return {
    shoulderTilt: Math.abs(signedShoulderTilt),
    headTilt: Math.abs(signedHeadTilt),
    signedShoulderTilt,
    signedHeadTilt,
    shoulderCenterX,
    noseCenterOffset,
    headShoulderOffsetX: (earCenter.x - shoulderCenter.x) / shoulderWidth,
    headShoulderOffsetY: (earCenter.y - shoulderCenter.y) / shoulderWidth,
    shoulderWidthRatio: shoulderWidth / video.videoWidth,
    shoulderHeightDifference:
      (Math.abs(leftShoulder.y - rightShoulder.y) / shoulderWidth) * 100,
    higherShoulder: leftShoulder.y < rightShoulder.y ? "Lijevo" : "Desno",
  };
}

function showLiveFlexion(flexion) {
  liveAngle.textContent = `${flexion.toFixed(1)}°`;
  liveAngle.hidden = false;
}

function showLiveLeg(landmarks, sample) {
  if (!landmarks) {
    liveAngle.hidden = true;
    setStatus("Osoba nije pronađena — prilagodi kadar ili osvjetljenje");
    return;
  }
  const side = frameSide(sample);
  drawPose(landmarks, side);
  const angle = side === "left" ? sample.angL : sample.angR;
  if (Number.isNaN(angle)) {
    liveAngle.hidden = true;
    setStatus("Odabrana noga nije dovoljno vidljiva", true);
    return;
  }
  recentFlexions.push(180 - angle);
  if (recentFlexions.length > SMOOTHING_WINDOW) recentFlexions.shift();
  showLiveFlexion(median(recentFlexions));
  setStatus(isRecording ? "Test se snima — zadrži nogu u kadru" : "Mjerenje uživo", true);
}

function showLivePosture(measurement) {
  liveAngle.textContent = `Ramena ${measurement.shoulderTilt.toFixed(1)}° · glava ${measurement.headTilt.toFixed(1)}°`;
  liveAngle.hidden = false;
}

function startPostureRepetition() {
  if (!cameraStream || postureGuideStep !== 2 || isRecording) return;
  if (postureRepResults.length >= POSTURE_REPETITIONS) {
    postureRepResults = [];
    clearResults();
  }
  recordingStats = createPostureStats();
  recordingStartedAt = performance.now();
  posturePhase = "recording";
  postureConditionSince = 0;
  postureStableMeasurements = [];
  isRecording = true;
  testMode.disabled = true;
  recordButton.disabled = true;
  const repetition = postureRepResults.length + 1;
  guideDemo.dataset.phase = "recording";
  guideDemoCaption.textContent = `Ponavljanje ${repetition} / ${POSTURE_REPETITIONS}`;
  recordButton.textContent = `Mjerenje ${repetition}/3 · 3 s`;
  resetButton.disabled = true;
  setGuideDetection(
    `Snimam ${repetition}/${POSTURE_REPETITIONS}. Ostani u udobnom položaju još 3 sekunde.`,
    true,
  );
  setStatus(`Snimam ponavljanje ${repetition}/3…`, true);
}

function finishPostureRepetition() {
  isRecording = false;
  const { samples, totalFrames } = recordingStats;
  if (samples.length < 5) {
    posturePhase = "stabilize";
    postureStableMeasurements = [];
    setGuideDetection("Nisam uhvatio dovoljno stabilnih mjerenja. Zadrži položaj da pokušam ponovno.");
    setStatus("Premalo vidljivih mjerenja — pokušaj ponovno");
    playGuideAnimation();
    return;
  }
  const higherLeftCount = samples.filter((sample) => sample.higherShoulder === "Lijevo").length;
  postureRepResults.push({
    shoulderTilt: median(samples.map((sample) => sample.shoulderTilt)),
    headTilt: median(samples.map((sample) => sample.headTilt)),
    shoulderHeightDifference: median(
      samples.map((sample) => sample.shoulderHeightDifference),
    ),
    higherShoulder:
      higherLeftCount === samples.length / 2
        ? "Nije stabilno"
        : higherLeftCount > samples.length / 2
          ? "Lijevo"
          : "Desno",
    visibility: Math.round((samples.length / totalFrames) * 100),
  });
  const completed = postureRepResults.length;
  const visibility = Math.round((samples.length / totalFrames) * 100);
  if (completed < POSTURE_REPETITIONS) {
    posturePhase = "wait-reset";
    postureResetBaseline = samples[Math.floor(samples.length / 2)];
    postureConditionSince = 0;
    postureStableMeasurements = [];
    guideDemo.dataset.phase = "wait-reset";
    guideDemoCaption.textContent = `Odmor · ${completed} / ${POSTURE_REPETITIONS}`;
    guideCopy.textContent =
      "Opusti se i malo promijeni položaj glave ili ramena. Kad kamera prepozna promjenu, vrati se u isti udoban položaj — sljedeće mjerenje počinje samo.";
    setGuideDetection(
      `Ponavljanje ${completed}/${POSTURE_REPETITIONS} gotovo (${visibility} % vidljivih). Čekam kratak opušteni pomak pa povratak u položaj.`,
      true,
    );
    resultNote.textContent =
      `Ponavljanje ${completed}/3 zabilježeno. Ponovi još ${POSTURE_REPETITIONS - completed} puta za sažetak i raspon ponovljivosti.`;
  } else {
    posturePhase = "complete";
    testMode.disabled = false;
    guideDemo.dataset.phase = "complete";
    guideDemoCaption.textContent = "Tri ponavljanja završena";
    guideTitle.textContent = "Mjerenje gotovo";
    guideStepCount.textContent = "GOTOVO";
    guideCopy.textContent = "Sažetak i pojedinačna ponavljanja prikazani su ispod.";
    setGuideDetection(
      `Sva 3 ponavljanja su gotova (${visibility} % vidljivih frameova u zadnjem).`,
      true,
    );
    resetButton.disabled = false;
    showResults(recordingStats);
  }
  setStatus(`Ponavljanje ${completed}/3 spremljeno`, true);
  playGuideAnimation();
}

function isPostureFramed(measurement) {
  return (
    measurement.shoulderWidthRatio >= 0.1 &&
    measurement.shoulderWidthRatio <= 0.65 &&
    measurement.shoulderCenterX >= 0.2 &&
    measurement.shoulderCenterX <= 0.8
  );
}

function isPostureFrontFacing(measurement) {
  return (
    isPostureFramed(measurement) &&
    Math.abs(measurement.noseCenterOffset) <= 0.35 &&
    measurement.shoulderTilt <= 15 &&
    measurement.headTilt <= 18
  );
}

function rememberPostureMeasurement(measurement, now) {
  postureStableMeasurements.push({ ...measurement, time: now });
  postureStableMeasurements = postureStableMeasurements.filter(
    (sample) => now - sample.time <= 800,
  );
}

function isPostureStable(now) {
  const samples = postureStableMeasurements;
  if (samples.length < 6 || samples[samples.length - 1].time - samples[0].time < 500) {
    return false;
  }
  const range = (key) => {
    const values = samples.map((sample) => sample[key]);
    return Math.max(...values) - Math.min(...values);
  };
  return (
    range("signedShoulderTilt") <= 5 &&
    range("signedHeadTilt") <= 5 &&
    range("headShoulderOffsetX") <= 0.07 &&
    range("headShoulderOffsetY") <= 0.07
  );
}

function postureHasReset(measurement) {
  if (!postureResetBaseline) return false;
  return (
    Math.abs(measurement.headShoulderOffsetX - postureResetBaseline.headShoulderOffsetX) >= 0.08 ||
    Math.abs(measurement.headShoulderOffsetY - postureResetBaseline.headShoulderOffsetY) >= 0.08 ||
    Math.abs(measurement.signedShoulderTilt - postureResetBaseline.signedShoulderTilt) >= 5 ||
    Math.abs(measurement.signedHeadTilt - postureResetBaseline.signedHeadTilt) >= 5
  );
}

function setGuideDetection(message, detected = false) {
  guideProgress.textContent = message;
  guideDetectionDot.classList.toggle("is-active", detected);
}

function updatePostureGuide(measurement, now) {
  if (posturePhase === "complete") return;
  if (!measurement) {
    postureConditionSince = 0;
    if (posturePhase !== "wait-reset") postureStableMeasurements = [];
    setGuideDetection("Tražim glavu i oba ramena — prilagodi kadar.");
    setStatus("Glava, uši i oba ramena moraju biti vidljivi");
    return;
  }
  if (posturePhase === "find-person") {
    if (!isPostureFramed(measurement)) {
      postureConditionSince = 0;
      setGuideDetection("Stani u sredinu kadra. Trebam vidjeti glavu i oba ramena.");
      return;
    }
    if (!postureConditionSince) postureConditionSince = now;
    const remaining = Math.max(0, SETUP_HOLD_MS - (now - postureConditionSince));
    setGuideDetection(
      remaining > 0
        ? `Kadar je pronađen — drži položaj još ${Math.ceil(remaining / 100) / 10} s.`
        : "Kadar prepoznat.",
      true,
    );
    if (remaining === 0) {
      posturePhase = "find-front";
      postureConditionSince = 0;
      postureStableMeasurements = [];
      setPostureGuideStep(1);
      setGuideDetection("Kadar je dobar. Sada gledaj ravno prema kameri.", true);
    }
    return;
  }
  if (posturePhase === "find-front") {
    if (!isPostureFrontFacing(measurement)) {
      postureConditionSince = 0;
      const instruction =
        Math.abs(measurement.noseCenterOffset) > 0.35
          ? "Okreni lice i tijelo prema kameri."
          : measurement.shoulderTilt > 15
            ? "Poravnaj telefon i ramena prema kameri."
            : "Namjesti glavu uspravno i gledaj u kameru.";
      setGuideDetection(instruction);
      return;
    }
    if (!postureConditionSince) postureConditionSince = now;
    const remaining = Math.max(0, FRONT_HOLD_MS - (now - postureConditionSince));
    setGuideDetection(
      remaining > 0
        ? `Pogled sprijeda prepoznat — ostani tako ${Math.ceil(remaining / 100) / 10} s.`
        : "Pogled sprijeda prepoznat.",
      true,
    );
    if (remaining === 0) {
      posturePhase = "stabilize";
      postureConditionSince = 0;
      postureStableMeasurements = [];
      setPostureGuideStep(2);
      guideCopy.textContent =
        "Položaj sprijeda prepoznat. Ostani opušteno i mirno; mjerenje će početi samo kad se položaj stabilizira.";
      setGuideDetection("Položaj prepoznat — čekam stabilnost za prvo mjerenje.", true);
    }
    return;
  }
  if (posturePhase === "wait-reset") {
    if (postureHasReset(measurement)) {
      posturePhase = "reacquire";
      postureConditionSince = 0;
      postureStableMeasurements = [];
      guideDemo.dataset.phase = "reacquire";
      guideDemoCaption.textContent = "Pomak prepoznat — vrati se u obris";
      guideCopy.textContent =
        "Pomak je zabilježen. Vrati glavu u sredinu obrisa, opusti ramena i ostani mirno — sljedeće mjerenje počinje automatski.";
      setGuideDetection("Pomak prepoznat. Čekam povratak u stabilan položaj.", true);
    } else {
      setGuideDetection("Opusti se i malo promijeni položaj glave ili ramena.");
    }
    return;
  }
  if (posturePhase === "reacquire" || posturePhase === "stabilize") {
    if (!isPostureFrontFacing(measurement)) {
      postureConditionSince = 0;
      postureStableMeasurements = [];
      setGuideDetection(
        posturePhase === "reacquire"
          ? "Vrati se u obris i gledaj ravno prema kameri."
          : "Gledaj ravno prema kameri i ostani u udobnom položaju.",
      );
      return;
    }
    rememberPostureMeasurement(measurement, now);
    if (!isPostureStable(now)) {
      postureConditionSince = 0;
      setGuideDetection("Položaj prepoznat — još malo mirno pričekaj.", true);
      return;
    }
    if (!postureConditionSince) postureConditionSince = now;
    const remaining = Math.max(0, NEUTRAL_HOLD_MS - (now - postureConditionSince));
    setGuideDetection(
      remaining > 0
        ? `Stabilan položaj — počinjem za ${Math.ceil(remaining / 100) / 10} s.`
        : "Stabilnost potvrđena — mjerenje počinje.",
      true,
    );
    if (remaining === 0) startPostureRepetition();
  }
}

function processFrame() {
  if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && landmarker) {
    resizeOverlay();
    if (video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      const result = landmarker.detectForVideo(video, nextDetectTimestamp());
      context.clearRect(0, 0, canvas.width, canvas.height);
      const landmarks = result.landmarks?.[0];
      if (isLegMode()) {
        const sample = legSample(
          landmarks,
          video.videoWidth,
          video.videoHeight,
          (performance.now() - recordingStartedAt) / 1000,
          MIN_VISIBILITY,
        );
        if (isRecording) legSamples.push(sample);
        showLiveLeg(landmarks, sample);
      } else if (landmarks) {
        const measurement = postureMeasurement(landmarks);
        if (measurement) {
          context.save();
          context.strokeStyle = "#b7f36b";
          context.lineWidth = Math.max(3, video.videoWidth / 300);
          for (const [first, second] of [[11, 12], [7, 8]]) {
            context.beginPath();
            context.moveTo(landmarks[first].x * canvas.width, landmarks[first].y * canvas.height);
            context.lineTo(landmarks[second].x * canvas.width, landmarks[second].y * canvas.height);
            context.stroke();
          }
          context.restore();
          showLivePosture(measurement);
          if (isRecording) recordingStats.samples.push(measurement);
          setStatus(
            posturePhase === "recording"
              ? "Test se snima — gledaj ravno i opusti ramena"
              : "Praćenje vodiča uživo",
            true,
          );
        } else {
          liveAngle.hidden = true;
          setStatus("Gledaj ravno prema kameri; oba ramena i uši moraju biti vidljivi");
        }
        updatePostureGuide(measurement, performance.now());
      } else {
        liveAngle.hidden = true;
        updatePostureGuide(null, performance.now());
      }
      if (isRecording && !isLegMode()) recordingStats.totalFrames += 1;
    }
  }
  if (
    isRecording &&
    testMode.value === "posture" &&
    performance.now() - recordingStartedAt >= POSTURE_REPETITION_MS
  ) {
    finishPostureRepetition();
  }
  animationFrame = requestAnimationFrame(processFrame);
}

function showResults(stats) {
  if (testMode.value === "posture") {
    if (postureRepResults.length < POSTURE_REPETITIONS) {
      resultNote.textContent = "Za sažetak snimi sva 3 ponavljanja.";
      return;
    }
    const headTilts = postureRepResults.map((sample) => sample.headTilt);
    const shoulderDifferences = postureRepResults.map(
      (sample) => sample.shoulderHeightDifference,
    );
    const headTilt = median(headTilts);
    const shoulderHeightDifference = median(shoulderDifferences);
    const range = (values) => Math.max(...values) - Math.min(...values);
    const leftHigherCount = postureRepResults.filter(
      (sample) => sample.higherShoulder === "Lijevo",
    ).length;
    const rightHigherCount = postureRepResults.filter(
      (sample) => sample.higherShoulder === "Desno",
    ).length;
    const higherShoulder =
      leftHigherCount === postureRepResults.length ||
      rightHigherCount === postureRepResults.length
        ? leftHigherCount > 0
          ? "lijevo rame je bilo više u sva tri ponavljanja"
          : "desno rame je bilo više u sva tri ponavljanja"
        : "strana višeg ramena nije bila ista u sva tri ponavljanja";
    metricOutputs[0].innerHTML = `${shoulderHeightDifference.toFixed(1)}<small>%</small>`;
    metricOutputs[1].innerHTML = `${headTilt.toFixed(1)}<small>°</small>`;
    metricOutputs[2].innerHTML = `${range(shoulderDifferences).toFixed(1)}<small>pp</small>`;
    repetitionDetails.replaceChildren(
      ...postureRepResults.map((sample, index) => {
        const item = document.createElement("li");
        item.textContent =
          `Ponavljanje ${index + 1} · ramena ${sample.shoulderHeightDifference.toFixed(1)} % · uši ${sample.headTilt.toFixed(1)}° · vidljivost ${sample.visibility} %`;
        return item;
      }),
    );
    repetitionDetails.hidden = false;
    resultNote.textContent =
      `Medijan 3 ponavljanja. ${higherShoulder}. Raspon nagiba linije ušiju: ${range(headTilts).toFixed(1)}°. Asimetrija = visinska razlika podijeljena širinom ramena; pp = postotni bodovi između pokušaja.`;
    return;
  }
  showLegResults();
}

const SIDE_NAMES = { left: "Lijeva noga", right: "Desna noga", near: "Noga bliža kameri" };

function sourceLabel() {
  return legSource === "kamera" ? "kamera uživo" : `snimka ${legSource}`;
}

/** Rezultat koljena ili hoda iz prikupljenih uzoraka — isti izračun kao knee_angle.py. */
function showLegResults() {
  const samples = legSamples;
  const coverage = analyzeKnee(samples, "near");
  if (coverage.validFrames < 5) {
    resultNote.textContent =
      `Nema dovoljno vidljivih mjerenja (${coverage.validFrames} frameova). Ponovi test s cijelom nogom u kadru.`;
    return;
  }
  const duration = samples[samples.length - 1].t - samples[0].t;
  if (testMode.value === "gait") {
    const gait = analyzeGait(samples);
    const strides = Object.values(gait?.legs ?? {}).filter((leg) => leg.strideS);
    if (!gait || !strides.length) {
      resultNote.textContent =
        "Nisam našao bar dva zamaha iste noge. Snimi dulji hod (barem 4–5 koraka) bočno prema kameri.";
      return;
    }
    const strideS = median(strides.map((leg) => leg.strideS));
    metricOutputs[0].innerHTML = `${gait.cadence.toFixed(0)}<small>/min</small>`;
    metricOutputs[1].innerHTML = `${strideS.toFixed(2)}<small>s</small>`;
    metricOutputs[2].innerHTML = `${gait.steps}<small>kor.</small>`;
    repetitionDetails.replaceChildren(
      ...Object.entries(gait.legs).map(([name, leg]) => {
        const item = document.createElement("li");
        item.textContent =
          `${name === "L" ? "Lijeva" : "Desna"} · ${leg.peaks.length} zamaha · fleksija u zamahu medijan ` +
          `${leg.flexion.length ? median(leg.flexion).toFixed(1) : "—"}° · ciklus ${leg.strideS ? `${leg.strideS.toFixed(2)} s` : "—"}`;
        return item;
      }),
    );
    repetitionDetails.hidden = false;
    resultNote.textContent =
      `Kadenca iz medijana trajanja ciklusa (2 koraka po ciklusu) · ${duration.toFixed(1)} s · ${sourceLabel()}. ` +
      "Brzina hoda se ne računa (treba statičnu kameru i visinu osobe).";
    currentMeasurement = {
      test: "hod",
      side: "obje",
      appValue: gait.cadence,
      unit: "koraka/min",
      cycleS: strideS,
      steps: gait.steps,
      visible: Math.round((coverage.validFrames / coverage.totalFrames) * 100),
    };
    drawResultChart(
      Object.entries(gait.legs).map(([name, leg]) => {
        const angles = samples.map((sample) => (name === "L" ? sample.angL : sample.angR));
        return {
          side: name === "L" ? "left" : "right",
          values: medianSmooth(angles).map((angle) => 180 - angle),
          peaks: leg.peaks,
        };
      }),
      "Fleksija obaju koljena kroz vrijeme",
    );
  } else {
    const side = sideSelect.value;
    const knee = analyzeKnee(samples, side);
    if (knee.validFrames < 5) {
      resultNote.textContent =
        `${SIDE_NAMES[side]} je vidljiva u samo ${knee.validFrames} frameova. Odaberi drugu nogu ili ponovi test.`;
      return;
    }
    metricOutputs[0].innerHTML = `${knee.minFlexion.toFixed(1)}<small>°</small>`;
    metricOutputs[1].innerHTML = `${knee.maxFlexion.toFixed(1)}<small>°</small>`;
    metricOutputs[2].innerHTML = `${knee.rom.toFixed(1)}<small>°</small>`;
    const percentage = Math.round((knee.validFrames / knee.totalFrames) * 100);
    const nearShare =
      side === "near" ? ` (lijeva ${Math.round(knee.nearLeftShare * 100)} % frameova)` : "";
    resultNote.textContent =
      `${SIDE_NAMES[side]}${nearShare} · ${knee.validFrames}/${knee.totalFrames} frameova (${percentage} % vidljivih) · ` +
      `${sourceLabel()} · treperenje ${knee.jitter.toFixed(1)}°/frame. Fleksija max usporedi s goniometrom.`;
    repetitionDetails.replaceChildren(
      ...knee.repetitions.map((repetition, index) => {
        const item = document.createElement("li");
        item.textContent =
          `Ponavljanje ${index + 1} · fleksija ${repetition.flexion.toFixed(1)}° · na ${repetition.t.toFixed(1)} s`;
        return item;
      }),
    );
    repetitionDetails.hidden = knee.repetitions.length === 0;
    currentMeasurement = {
      test: "koljeno",
      side: { left: "lijeva", right: "desna", near: "bliža" }[side],
      appValue: knee.maxFlexion,
      unit: "° fleksije",
      rom: knee.rom,
      repetitions: knee.repetitions.length,
      visible: percentage,
    };
    const peakIndices = knee.repetitions.map((repetition) => knee.times.indexOf(repetition.t));
    drawResultChart(
      [{ side, values: knee.flexion, peaks: peakIndices }],
      `Fleksija — ${SIDE_NAMES[side].toLowerCase()}`,
    );
  }
  currentMeasurement.source = sourceLabel();
  currentMeasurement.durationS = duration;
  saveForm.hidden = false;
  goniometerField.hidden = testMode.value === "gait";
  resetButton.disabled = false;
}

function drawResultChart(series, title) {
  chartData = { times: legSamples.map((sample) => sample.t), series, hover: null };
  chartTitle.textContent = title;
  chartLegend.hidden = series.length < 2 && !series.some((item) => item.peaks.length);
  chartLegend.querySelector(".swatch-left").parentElement.hidden = !series.some((item) => item.side === "left");
  chartLegend.querySelector(".swatch-right").parentElement.hidden = !series.some((item) => item.side === "right");
  chart.hidden = false;
  renderChart();
}

/** Linijski graf fleksije (°) kroz vrijeme (s); vrhovi ponavljanja označeni, hover pokazuje vrijednosti. */
function renderChart() {
  if (!chartData || chart.hidden) return;
  const ratio = window.devicePixelRatio || 1;
  const width = chartCanvas.clientWidth;
  const height = chartCanvas.clientHeight;
  if (!width || !height) return;
  chartCanvas.width = Math.round(width * ratio);
  chartCanvas.height = Math.round(height * ratio);
  const g = chartCanvas.getContext("2d");
  g.setTransform(ratio, 0, 0, ratio, 0, 0);
  g.clearRect(0, 0, width, height);

  const { times, series, hover } = chartData;
  const pad = { left: 34, right: 10, top: 10, bottom: 22 };
  const t0 = times[0];
  const t1 = times[times.length - 1] || t0 + 1;
  const allValues = series.flatMap((item) => item.values.filter(Number.isFinite));
  const yMax = Math.max(90, Math.ceil(Math.max(...allValues) / 30) * 30);
  const yMin = Math.min(0, Math.floor(Math.min(...allValues) / 30) * 30);
  const x = (t) => pad.left + ((t - t0) / (t1 - t0 || 1)) * (width - pad.left - pad.right);
  const y = (value) => pad.top + ((yMax - value) / (yMax - yMin)) * (height - pad.top - pad.bottom);

  g.font = "11px Inter, system-ui, sans-serif";
  g.fillStyle = "#82928a";
  g.strokeStyle = "rgba(202, 228, 211, 0.08)";
  g.lineWidth = 1;
  g.textAlign = "right";
  g.textBaseline = "middle";
  for (let value = yMin; value <= yMax; value += 30) {
    g.beginPath();
    g.moveTo(pad.left, y(value));
    g.lineTo(width - pad.right, y(value));
    g.stroke();
    g.fillText(`${value}°`, pad.left - 6, y(value));
  }
  g.textAlign = "center";
  g.textBaseline = "top";
  const tickStep = Math.max(1, Math.ceil((t1 - t0) / 6));
  for (let t = Math.ceil(t0); t <= t1; t += tickStep) {
    g.fillText(`${t} s`, x(t), height - pad.bottom + 6);
  }

  for (const item of series) {
    g.strokeStyle = LEG_COLORS[item.side];
    g.lineWidth = 2;
    g.lineJoin = "round";
    g.beginPath();
    let drawing = false;
    item.values.forEach((value, index) => {
      if (!Number.isFinite(value)) {
        drawing = false;
        return;
      }
      if (drawing) g.lineTo(x(times[index]), y(value));
      else g.moveTo(x(times[index]), y(value));
      drawing = true;
    });
    g.stroke();
    for (const index of item.peaks) {
      // trokut prema dolje na vrhu, s prstenom boje podloge
      const px = x(times[index]);
      const py = y(item.values[index]) - 9;
      g.beginPath();
      g.moveTo(px - 5, py - 4);
      g.lineTo(px + 5, py - 4);
      g.lineTo(px, py + 4);
      g.closePath();
      g.fillStyle = LEG_COLORS[item.side];
      g.strokeStyle = "#14221f";
      g.lineWidth = 2;
      g.stroke();
      g.fill();
    }
  }

  if (hover !== null) {
    const px = x(times[hover]);
    g.strokeStyle = "rgba(242, 245, 239, 0.45)";
    g.lineWidth = 1;
    g.beginPath();
    g.moveTo(px, pad.top);
    g.lineTo(px, height - pad.bottom);
    g.stroke();
    for (const item of series) {
      const value = item.values[hover];
      if (!Number.isFinite(value)) continue;
      g.beginPath();
      g.arc(px, y(value), 4, 0, Math.PI * 2);
      g.fillStyle = LEG_COLORS[item.side];
      g.strokeStyle = "#14221f";
      g.lineWidth = 2;
      g.stroke();
      g.fill();
    }
  }
  chartData.layout = { x, pad, width };
}

function updateChartHover(event) {
  if (!chartData?.layout) return;
  const { times, series } = chartData;
  const bounds = chartCanvas.getBoundingClientRect();
  const pointerX = event.clientX - bounds.left;
  let nearest = 0;
  let distance = Infinity;
  times.forEach((time, index) => {
    const gap = Math.abs(chartData.layout.x(time) - pointerX);
    if (gap < distance) {
      distance = gap;
      nearest = index;
    }
  });
  chartData.hover = nearest;
  renderChart();
  const values = series
    .map((item) => {
      const value = item.values[nearest];
      const name = { left: "lijeva", right: "desna", near: "bliža" }[item.side];
      return `${series.length > 1 ? `${name} ` : ""}${Number.isFinite(value) ? `${value.toFixed(1)}°` : "—"}`;
    })
    .join(" · ");
  chartTooltip.textContent = `${times[nearest].toFixed(2)} s · ${values}`;
  chartTooltip.hidden = false;
  const left = Math.min(Math.max(chartData.layout.x(times[nearest]), 70), bounds.width - 70);
  chartTooltip.style.left = `${left}px`;
}

function hideChartHover() {
  if (!chartData) return;
  chartData.hover = null;
  chartTooltip.hidden = true;
  renderChart();
}

function readHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
  } catch {
    return [];
  }
}

function writeHistory(entries) {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(entries));
    return true;
  } catch (error) {
    console.error("Spremanje nije uspjelo:", error);
    return false;
  }
}

function formatNumber(value, digits = 1) {
  return Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function renderHistory() {
  const entries = readHistory();
  exportButton.disabled = entries.length === 0;
  clearHistoryButton.disabled = entries.length === 0;
  historyTable.hidden = entries.length === 0;
  historyBody.replaceChildren(
    ...entries
      .slice()
      .reverse()
      .map((entry) => {
        const row = document.createElement("tr");
        const difference = entry.difference;
        const cells = [
          new Date(entry.date).toLocaleString("hr-HR", { dateStyle: "short", timeStyle: "short" }),
          entry.patient || "—",
          entry.test === "hod" ? "Hod" : `Koljeno · ${entry.side}`,
          entry.source,
          entry.test === "hod" ? `${formatNumber(entry.appValue, 0)}/min` : `${formatNumber(entry.appValue)}°`,
          Number.isFinite(entry.goniometer) ? `${formatNumber(entry.goniometer)}°` : "—",
          Number.isFinite(difference) ? `${difference > 0 ? "+" : ""}${formatNumber(difference)}°` : "—",
        ];
        cells.forEach((text, index) => {
          const cell = document.createElement("td");
          cell.textContent = text;
          if (index >= 4) cell.className = "num";
          if (index === 6 && Number.isFinite(difference)) {
            cell.classList.add(Math.abs(difference) <= AGREEMENT_LIMIT ? "is-within" : "is-outside");
            cell.textContent += Math.abs(difference) <= AGREEMENT_LIMIT ? " ✓" : " ✕";
          }
          row.append(cell);
        });
        return row;
      }),
  );
  const compared = entries.filter((entry) => Number.isFinite(entry.difference));
  if (!entries.length) {
    historySummary.textContent = "Još nema spremljenih mjerenja.";
  } else if (!compared.length) {
    historySummary.textContent =
      `${entries.length} mjerenja. Upiši goniometar uz mjerenje koljena za usporedbu (cilj PoC-a ±${AGREEMENT_LIMIT}°).`;
  } else {
    const within = compared.filter((entry) => Math.abs(entry.difference) <= AGREEMENT_LIMIT).length;
    const differences = compared.map((entry) => entry.difference);
    const meanAbs = differences.reduce((sum, value) => sum + Math.abs(value), 0) / differences.length;
    const bias = differences.reduce((sum, value) => sum + value, 0) / differences.length;
    historySummary.textContent =
      `Unutar ±${AGREEMENT_LIMIT}°: ${within}/${compared.length} usporedbi s goniometrom · ` +
      `prosječna |razlika| ${meanAbs.toFixed(1)}° · prosječni pomak app − goniometar ${bias > 0 ? "+" : ""}${bias.toFixed(1)}°.`;
  }
}

function downloadText(filename, text) {
  const blob = new Blob([`﻿${text}`], { type: "text/csv;charset=utf-8" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

function fileStamp() {
  return new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-");
}

function once(target, eventName, timeoutMs = 8000) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      target.removeEventListener(eventName, onEvent);
      reject(new Error(`Isteklo čekanje na ${eventName}`));
    }, timeoutMs);
    function onEvent() {
      clearTimeout(timer);
      resolve();
    }
    target.addEventListener(eventName, onEvent, { once: true });
  });
}

/**
 * Frameovi u sekundi snimke: preglednik ih ne daje izravno, pa se pusti kratak dio i izmjere
 * razmaci prikazanih frameova. Najmanji razmak (preskočeni frameovi daju veće), zaokružen na standardni fps.
 */
async function detectFrameRate() {
  if (!("requestVideoFrameCallback" in HTMLVideoElement.prototype)) return 30;
  const mediaTimes = [];
  await new Promise((resolve) => {
    const stopAt = performance.now() + 2000;
    const onFrame = (_now, metadata) => {
      mediaTimes.push(metadata.mediaTime);
      if (mediaTimes.length >= 15 || performance.now() > stopAt) resolve();
      else video.requestVideoFrameCallback(onFrame);
    };
    video.requestVideoFrameCallback(onFrame);
    video.play().catch(resolve);
    setTimeout(resolve, 2500);
  });
  video.pause();
  const steps = mediaTimes
    .slice(1)
    .map((time, index) => time - mediaTimes[index])
    .filter((step) => step > 0.004);
  if (steps.length < 3) return 30;
  const measured = 1 / Math.min(...steps);
  const nearest = COMMON_FRAME_RATES.reduce((best, rate) =>
    Math.abs(rate - measured) < Math.abs(best - measured) ? rate : best,
  );
  return Math.abs(nearest - measured) / nearest < 0.05 ? nearest : measured;
}

/** Snimka iz galerije: svaki frame ide kroz model, kao knee_angle.py nad datotekom. */
async function analyzeVideoFile(file) {
  if (cameraStream) stopCamera();
  fileAnalysis = { cancelled: false };
  fileButton.textContent = "Učitavam model…";
  cameraButton.disabled = true;
  recordButton.disabled = true;
  testMode.disabled = true;
  sideSelect.disabled = true;
  const url = URL.createObjectURL(file);
  try {
    await ensureLandmarker();
    clearResults();
    video.srcObject = null;
    viewer.classList.add("is-live");
    viewerMessage.hidden = true;
    video.preload = "auto";
    const metadata = once(video, "loadedmetadata");
    video.src = url;
    await metadata;
    const duration = video.duration;
    if (!Number.isFinite(duration) || duration <= 0) throw new Error("Ne mogu pročitati trajanje snimke.");
    setStatus("Određujem broj frameova u sekundi…", true);
    const fps = await detectFrameRate();
    const frames = Math.floor(duration * fps);
    const samples = [];
    recentFlexions = [];
    fileButton.textContent = "Prekini analizu";
    for (let frame = 0; frame < frames && !fileAnalysis.cancelled; frame += 1) {
      const t = frame / fps;
      // malo iza početka framea da preglednik ne zaokruži na prethodni
      video.currentTime = t + 0.001;
      await once(video, "seeked");
      resizeOverlay();
      const result = landmarker.detectForVideo(video, nextDetectTimestamp());
      const landmarks = result.landmarks?.[0];
      const sample = legSample(landmarks, video.videoWidth, video.videoHeight, t, MIN_VISIBILITY);
      samples.push(sample);
      context.clearRect(0, 0, canvas.width, canvas.height);
      showLiveLeg(landmarks, sample);
      setStatus(`Analiziram ${file.name}: ${Math.round(((frame + 1) / frames) * 100)} %`, true);
    }
    if (fileAnalysis.cancelled) {
      setStatus("Analiza snimke prekinuta");
      return;
    }
    legSamples = samples;
    legSource = file.name;
    setStatus(`Snimka analizirana: ${samples.length} frameova (${duration.toFixed(1)} s, ${fps.toFixed(2)} fps)`, true);
    showLegResults();
  } catch (error) {
    console.error("Analiza snimke nije uspjela:", error);
    setStatus(error instanceof Error ? error.message : "Analiza snimke nije uspjela.");
  } finally {
    fileAnalysis = undefined;
    URL.revokeObjectURL(url);
    fileButton.textContent = "Analiziraj snimku iz galerije";
    cameraButton.disabled = false;
    testMode.disabled = false;
    sideSelect.disabled = false;
    setBusy(false);
  }
}

function clearResults() {
  repetitionDetails.replaceChildren();
  repetitionDetails.hidden = true;
  metricOutputs.forEach((output) => {
    output.innerHTML = "—<small>°</small>";
  });
  recordingStats = createPostureStats();
  chart.hidden = true;
  chartData = undefined;
  chartTooltip.hidden = true;
  saveForm.hidden = true;
  currentMeasurement = undefined;
  if (testMode.value === "posture") {
    resultsTitle.textContent = "Držanje ramena i glave";
    metricLabels[0].textContent = "Asimetrija ramena";
    metricLabels[1].textContent = "Nagib linije ušiju";
    metricLabels[2].textContent = "Raspon ponavljanja";
    metricOutputs[0].innerHTML = "—<small>%</small>";
    metricOutputs[2].innerHTML = "—<small>pp</small>";
    resultNote.textContent =
      "Pokreni kameru. Vodič automatski prepoznaje kadar i položaj pa bilježi 3 ponavljanja.";
  } else if (testMode.value === "gait") {
    resultsTitle.textContent = "Kadenca hoda";
    metricLabels[0].textContent = "Kadenca";
    metricLabels[1].textContent = "Ciklus (medijan)";
    metricLabels[2].textContent = "Koraka";
    metricOutputs[0].innerHTML = "—<small>/min</small>";
    metricOutputs[1].innerHTML = "—<small>s</small>";
    metricOutputs[2].innerHTML = "—<small>kor.</small>";
    resultNote.textContent =
      "Analiziraj snimku hoda ili pokreni kameru i snimi barem 4–5 koraka bočno prema kameri.";
  } else {
    resultsTitle.textContent = "Opseg pokreta koljena";
    metricLabels[0].textContent = "Fleksija min";
    metricLabels[1].textContent = "Fleksija max";
    metricLabels[2].textContent = "ROM";
    resultNote.textContent = "Pokreni kameru i pritisni „Snimi test” ili analiziraj snimku iz galerije.";
  }
  resetButton.disabled = true;
}

function playGuideAnimation() {
  guideDemo.classList.remove("animate");
  requestAnimationFrame(() => guideDemo.classList.add("animate"));
}

function setPostureGuideStep(step) {
  postureGuideStep = step;
  const steps = [
    {
      title: "Postavi kameru",
      caption: "Telefon stabilno, u visini ramena",
      copy: "Postavi telefon ravno, približno u visini ramena. Stani dovoljno daleko da se vide cijela glava, vrat i oba ramena.",
    },
    {
      title: "Namjesti položaj",
      caption: "Pogled ravno, glava udobno centrirana",
      copy: "Okreni se ravno prema kameri. Drži glavu u udobnom, neutralnom položaju i opusti ramena — nemoj ih namjerno podizati ili poravnavati.",
    },
    {
      title: "Snimi 3 ponavljanja",
      caption: "Mirno stajanje · 3 sekunde",
      copy: "Ostani mirno u udobnom položaju. Snimanje počinje samo kad kamera prepozna stabilnost.",
    },
  ];
  const current = steps[step];
  guideTitle.textContent = current.title;
  guideStepCount.textContent = `KORAK ${step + 1} / 3`;
  guideDemoCaption.textContent = current.caption;
  guideCopy.textContent = current.copy;
  guideDemo.dataset.step = String(step);
  guideDemo.dataset.phase = step === 0 ? "find-person" : step === 1 ? "find-front" : "stabilize";
  guideDetectionDot.classList.remove("is-active");
  playGuideAnimation();
}

function updateTestMode() {
  const posture = testMode.value === "posture";
  const gait = testMode.value === "gait";
  if (cameraStream) stopCamera();
  legSamples = [];
  sideControls.hidden = posture || gait;
  fileControls.hidden = posture;
  postureGuide.hidden = !posture;
  appTitle.textContent = posture ? "Ramena i glava" : gait ? "Hod — kadenca" : "Mjerenje koljena";
  testHint.textContent = posture
    ? "Vodič automatski prati kadar i položaj. Mjerenje je opisno i ne određuje ispravno ili neispravno držanje."
    : gait
      ? "Telefon vodoravno i mirno, bočno od staze, 3–4 m daleko. Snimka iz galerije analizira svaki frame i pouzdanija je od mjerenja uživo."
      : "Telefon vodoravno, bočno prema nozi, cijela noga u kadru. Za validaciju je najbolja snimka iz galerije (svaki frame), uživo ovisi o brzini uređaja.";
  postureRepResults = [];
  posturePhase = "find-person";
  postureConditionSince = 0;
  postureStableMeasurements = [];
  postureResetBaseline = undefined;
  setPostureGuideStep(0);
  setGuideDetection("Pokreni kameru; vodič će sam prepoznati pripremne korake.");
  clearResults();
  recordButton.disabled = true;
  recordButton.textContent = "Snimi test";
  recordButton.hidden = posture;
  cameraButton.textContent = posture ? "Pokreni kameru i vodič" : "Pokreni kameru";
}

async function ensureLandmarker() {
  if (landmarker) return;
  setBusy(true);
  setStatus("Učitavam model za procjenu poze…");
  const vision = await FilesetResolver.forVisionTasks(WASM_URL);
  const options = {
    runningMode: "VIDEO",
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  };
  try {
    landmarker = await PoseLandmarker.createFromOptions(vision, {
      ...options,
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
    });
  } catch (gpuError) {
    console.warn("GPU nije dostupan; pokušavam CPU:", gpuError);
    landmarker = await PoseLandmarker.createFromOptions(vision, {
      ...options,
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "CPU" },
    });
  }
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("Kamera nije dostupna. Otvori stranicu preko HTTPS-a ili localhosta.");
  }
  try {
    await ensureLandmarker();
    setBusy(true);
    cameraStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: testMode.value === "posture" ? "user" : "environment",
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
    });
    video.removeAttribute("src");
    video.srcObject = cameraStream;
    await video.play();
    if (testMode.value === "posture") {
      postureRepResults = [];
      posturePhase = "find-person";
      postureConditionSince = 0;
      postureStableMeasurements = [];
      postureResetBaseline = undefined;
      setPostureGuideStep(0);
      setGuideDetection("Tražim glavu i oba ramena u kadru.");
      testMode.disabled = true;
    }
    viewer.classList.add("is-live");
    viewerMessage.hidden = true;
    cameraButton.textContent = "Zaustavi kameru";
    cameraButton.disabled = false;
    recordButton.disabled = false;
    if (testMode.value === "posture") {
      recordButton.hidden = true;
      recordButton.disabled = true;
      cameraButton.textContent = "Zaustavi kameru";
    }
    sideSelect.disabled = false;
    clearResults();
    setStatus(
      testMode.value === "posture"
        ? "Kamera je spremna — stani sprijeda i snimi test"
        : "Kamera je spremna — odaberi nogu i snimi test",
      true,
    );
    fileButton.disabled = true;
    animationFrame = requestAnimationFrame(processFrame);
  } catch (error) {
    cameraStream?.getTracks().forEach((track) => track.stop());
    cameraStream = undefined;
    setBusy(false);
    setStatus(error instanceof Error ? error.message : "Pokretanje kamere nije uspjelo.");
    throw error;
  }
}

function stopCamera() {
  cancelAnimationFrame(animationFrame);
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = undefined;
  }
  video.srcObject = null;
  video.removeAttribute("src");
  video.load();
  context.clearRect(0, 0, canvas.width, canvas.height);
  viewer.classList.remove("is-live");
  fileButton.disabled = false;
  viewerMessage.hidden = false;
  liveAngle.hidden = true;
  recordButton.disabled = true;
  recordButton.textContent = "Snimi test";
  recordButton.hidden = testMode.value === "posture";
  cameraButton.disabled = false;
  cameraButton.textContent =
    testMode.value === "posture" ? "Pokreni kameru i vodič" : "Pokreni kameru";
  sideSelect.disabled = false;
  isRecording = false;
  recentFlexions = [];
  testMode.disabled = false;
  setStatus("Kamera nije pokrenuta");
}

cameraButton.addEventListener("click", async () => {
  if (cameraStream) {
    stopCamera();
    return;
  }
  try {
    await startCamera();
  } catch (error) {
    console.error("Neuspjelo pokretanje kamere:", error);
  }
});

recordButton.addEventListener("click", () => {
  if (testMode.value === "posture") {
    startPostureRepetition();
    return;
  }
  if (!isRecording) {
    clearResults();
    legSamples = [];
    legSource = "kamera";
    recordingStartedAt = performance.now();
    recentFlexions = [];
    isRecording = true;
    testMode.disabled = true;
    sideSelect.disabled = true;
    recordButton.textContent = "Završi test";
    resetButton.disabled = true;
    resultNote.textContent = "Test je u tijeku…";
    return;
  }
  isRecording = false;
  testMode.disabled = false;
  sideSelect.disabled = false;
  recordButton.textContent = "Snimi novi test";
  showLegResults();
});

resetButton.addEventListener("click", () => {
  if (testMode.value === "posture") {
    postureRepResults = [];
    posturePhase = "find-person";
    postureConditionSince = 0;
    postureStableMeasurements = [];
    postureResetBaseline = undefined;
    setPostureGuideStep(0);
    setGuideDetection(
      cameraStream
        ? "Stani u sredinu kadra. Vodič će sam započeti novi test."
        : "Pokreni kameru; vodič će sam prepoznati pripremne korake.",
    );
  }
  clearResults();
});
sideSelect.addEventListener("change", () => {
  recentFlexions = [];
  liveAngle.hidden = true;
  clearResults();
  if (legSamples.length && !isRecording) showLegResults();
});

fileButton.addEventListener("click", () => {
  if (fileAnalysis) {
    fileAnalysis.cancelled = true;
    return;
  }
  fileInput.click();
});

fileInput.addEventListener("change", () => {
  const [file] = fileInput.files;
  fileInput.value = "";
  if (file) analyzeVideoFile(file);
});

chartCanvas.addEventListener("pointermove", updateChartHover);
chartCanvas.addEventListener("pointerleave", hideChartHover);
window.addEventListener("resize", renderChart);

saveForm.addEventListener("submit", (event) => {
  event.preventDefault();
  if (!currentMeasurement) return;
  const goniometer =
    testMode.value === "gait" || goniometerInput.value === ""
      ? NaN
      : Number(goniometerInput.value.replace(",", "."));
  const entry = {
    ...currentMeasurement,
    id: crypto.randomUUID?.() ?? String(Date.now()),
    date: new Date().toISOString(),
    patient: patientInput.value.trim(),
    goniometer,
    difference: Number.isFinite(goniometer) ? currentMeasurement.appValue - goniometer : NaN,
  };
  const entries = readHistory();
  entries.push(entry);
  if (!writeHistory(entries)) {
    resultNote.textContent = "Spremanje nije uspjelo (preglednik ne dopušta lokalnu pohranu).";
    return;
  }
  goniometerInput.value = "";
  saveForm.hidden = true;
  resultNote.textContent += " Spremljeno.";
  renderHistory();
});

framesButton.addEventListener("click", () => {
  if (!legSamples.length) return;
  const side = testMode.value === "gait" ? "near" : sideSelect.value;
  downloadText(`fizio-frameovi-${fileStamp()}.csv`, samplesCsv(legSamples, side));
});

exportButton.addEventListener("click", () => {
  const entries = readHistory();
  const csv = toCsv(
    [
      "datum", "oznaka", "test", "noga", "izvor", "app_vrijednost", "jedinica", "goniometar",
      "razlika_app_minus_gonio", "rom", "ponavljanja", "kadenca_ciklus_s", "koraka", "vidljivo_posto",
      "trajanje_s",
    ],
    entries.map((entry) => [
      entry.date, entry.patient, entry.test, entry.side, entry.source, entry.appValue, entry.unit,
      entry.goniometer, entry.difference, entry.rom, entry.repetitions, entry.cycleS, entry.steps,
      entry.visible, entry.durationS,
    ]),
  );
  downloadText(`fizio-mjerenja-${fileStamp()}.csv`, csv);
});

clearHistoryButton.addEventListener("click", () => {
  if (!confirm("Obrisati sva spremljena mjerenja s ovog uređaja?")) return;
  writeHistory([]);
  renderHistory();
});
testMode.addEventListener("change", () => {
  recentFlexions = [];
  liveAngle.hidden = true;
  updateTestMode();
});
updateTestMode();
renderHistory();

window.addEventListener("beforeinstallprompt", (event) => {
  event.preventDefault();
  installPrompt = event;
  installButton.hidden = false;
});

installButton.addEventListener("click", async () => {
  if (!installPrompt) return;
  await installPrompt.prompt();
  installPrompt = undefined;
  installButton.hidden = true;
});

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("./sw.js").catch((error) => {
      console.error("Registracija service workera nije uspjela:", error);
    });
  });
}
