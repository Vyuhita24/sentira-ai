const recordBtn = document.getElementById("record-btn");
const recordingStatus = document.getElementById("recording-status");
const fileInput = document.getElementById("file-input");
const dropZone = document.getElementById("drop-zone");
const fileName = document.getElementById("file-name");
const resultBox = document.getElementById("result");
const emotionLabel = document.getElementById("emotion-label");
const stressStateBadge = document.getElementById("stress-state-badge");
const confidenceFill = document.getElementById("confidence-fill");
const confidenceValue = document.getElementById("confidence-value");
const vocalInsight = document.getElementById("vocal-insight");
const suggestedAction = document.getElementById("suggested-action");
const contactForm = document.getElementById("contact-form");

let mediaRecorder = null;
let micStream = null;
let audioChunks = [];
let isRecording = false;
let selectedFile = null;
let mediaRecorderMimeType = "";

function setText(element, value) {
    if (element) {
        element.textContent = value;
    }
}

function isWavFile(file) {
    if (!file) {
        return false;
    }

    const fileNameLower = (file.name || "").toLowerCase();
    const typeLower = (file.type || "").toLowerCase();
    return fileNameLower.endsWith(".wav") || typeLower.includes("wav");
}

function setSelectedFile(file) {
    if (!isWavFile(file)) {
        alert("Please upload a WAV audio file.");
        return;
    }

    selectedFile = file;
    setText(fileName, `Selected: ${file.name}`);
}

function openTab(tabName) {
    const tabs = document.querySelectorAll(".tab-content");
    const buttons = document.querySelectorAll(".tab-btn");

    tabs.forEach((tab) => {
        tab.classList.toggle("active", tab.id === tabName);
    });

    buttons.forEach((button) => {
        const target = button.getAttribute("onclick") || "";
        button.classList.toggle("active", target.includes(`'${tabName}'`));
    });

    hideResult();
}

async function toggleRecording() {
    if (isRecording) {
        stopRecording();
        return;
    }
    await startRecording();
}

async function startRecording() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert("Microphone access is not supported in this browser.");
        return;
    }

    try {
        micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorderMimeType = getSupportedRecorderMimeType();
        const recorderOptions = mediaRecorderMimeType ? { mimeType: mediaRecorderMimeType } : {};
        mediaRecorder = new MediaRecorder(micStream, recorderOptions);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = async () => {
            const recordedBlob = new Blob(audioChunks, {
                type: mediaRecorder.mimeType || mediaRecorderMimeType || "audio/webm"
            });

            if (micStream) {
                micStream.getTracks().forEach((track) => track.stop());
                micStream = null;
            }

            try {
                const wavBlob = await convertBlobToWav(recordedBlob);
                const wavFile = new File([wavBlob], "live_recording.wav", { type: "audio/wav" });
                await uploadFile(wavFile);
            } catch (conversionError) {
                console.warn("WAV conversion failed, using recorded format:", conversionError);
                const fallbackExtension = extensionFromMime(recordedBlob.type || mediaRecorderMimeType);
                const fallbackFile = new File(
                    [recordedBlob],
                    `live_recording${fallbackExtension}`,
                    { type: recordedBlob.type || "application/octet-stream" }
                );
                await uploadFile(fallbackFile);
            }
        };

        mediaRecorder.start();
        isRecording = true;
        if (recordBtn) {
            recordBtn.innerHTML = '<i class="fa-solid fa-stop"></i> STOP RECORDING';
            recordBtn.classList.add("recording");
        }
        setText(recordingStatus, "Recording... Speak now.");
    } catch (error) {
        console.error("Error accessing microphone:", error);
        alert("Could not access microphone.");
    }
}

function stopRecording() {
    if (!mediaRecorder || !isRecording) {
        return;
    }

    mediaRecorder.stop();
    isRecording = false;
    if (recordBtn) {
        recordBtn.innerHTML = '<i class="fa-solid fa-microphone"></i> START RECORDING';
        recordBtn.classList.remove("recording");
    }
    setText(recordingStatus, "Processing recording...");
}

function handleAnalyzeClick() {
    if (!selectedFile) {
        alert("Please select or drop a WAV audio file first.");
        return;
    }

    uploadFile(selectedFile);
}

async function uploadFile(file) {
    showLoading();

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("/predict_audio", {
            method: "POST",
            body: formData
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || "An error occurred during analysis.");
        }

        displayResult(data.result, data.confidence, data.emotion);
    } catch (error) {
        console.error("Upload error:", error);
        alert(error.message || "Failed to connect to the server.");
        hideResult();
    }
}

function showLoading() {
    if (resultBox) {
        resultBox.classList.remove("hidden");
    }

    setText(emotionLabel, "Analyzing...");
    setText(stressStateBadge, "STRESS STATE: PROCESSING");

    if (stressStateBadge) {
        stressStateBadge.classList.remove("low", "medium", "high");
        stressStateBadge.classList.add("medium");
    }

    if (confidenceFill) {
        confidenceFill.style.width = "0%";
        confidenceFill.style.background = "linear-gradient(90deg, #ffc85b, #f49d3e)";
    }
    setText(confidenceValue, "0%");
    setText(vocalInsight, "Extracting acoustic features and evaluating emotional markers...");
    setText(suggestedAction, "Please wait while the model processes your audio sample.");
}

function displayResult(result, confidence, emotion) {
    if (resultBox) {
        resultBox.classList.remove("hidden");
    }

    const displayEmotion = formatEmotion(emotion || result || "Neutral");
    const stressMeta = resolveStressMeta(result, displayEmotion);

    setText(emotionLabel, displayEmotion);
    setText(stressStateBadge, `STRESS STATE: ${stressMeta.label.toUpperCase()}`);

    if (stressStateBadge) {
        stressStateBadge.classList.remove("low", "medium", "high");
        stressStateBadge.classList.add(stressMeta.level);
    }

    const numericConfidence = Number(confidence);
    const scaledConfidence = Number.isFinite(numericConfidence)
        ? (numericConfidence <= 1 ? numericConfidence * 100 : numericConfidence)
        : 0;
    const boundedConfidence = Math.max(0, Math.min(99.9, scaledConfidence));
    const displayConfidence = `${boundedConfidence.toFixed(1)}%`;

    if (confidenceFill) {
        confidenceFill.style.width = `${boundedConfidence}%`;
        confidenceFill.style.background = stressMeta.gradient;
    }
    setText(confidenceValue, displayConfidence);

    setText(vocalInsight, buildVocalInsight(stressMeta.level, displayEmotion, boundedConfidence));
    setText(suggestedAction, buildSuggestedAction(stressMeta.level, displayEmotion));

    if (recordingStatus && !isRecording) {
        setText(recordingStatus, "Ready to capture audio stream...");
    }
}

function hideResult() {
    if (resultBox) {
        resultBox.classList.add("hidden");
    }
}

function formatEmotion(value) {
    const map = {
        hap: "Happy",
        happy: "Happy",
        neu: "Neutral",
        neutral: "Neutral",
        ang: "Angry",
        angry: "Angry",
        sad: "Sad",
        fear: "Fearful",
        fearful: "Fearful",
        disgust: "Disgust",
        surprised: "Surprised",
        surprise: "Surprised",
        pleasant_surprise: "Pleasant Surprise",
        calm: "Calm"
    };

    const key = String(value || "").trim().toLowerCase();
    return map[key] || String(value || "Neutral").replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function resolveStressMeta(result, emotion) {
    const value = String(result || "").toLowerCase();
    const pleasant = ["Happy", "Neutral", "Calm", "Surprised", "Pleasant Surprise"];

    if (value.includes("not stressed") || value.includes("no stress")) {
        return { level: "low", label: "Low", gradient: "linear-gradient(90deg, #3ce6ff, #4474ff)" };
    }

    if (value.includes("high") || value.includes("severe")) {
        return { level: "high", label: "High", gradient: "linear-gradient(90deg, #ff8d8d, #ff4a6f)" };
    }

    if (value.includes("medium") || value.includes("moderate")) {
        return { level: "medium", label: "Medium", gradient: "linear-gradient(90deg, #ffc85b, #f49d3e)" };
    }

    if (value.includes("stressed") || value.includes("stress")) {
        return { level: "high", label: "High", gradient: "linear-gradient(90deg, #ff8d8d, #ff4a6f)" };
    }

    if (value.includes("low") || value.includes("calm")) {
        return { level: "low", label: "Low", gradient: "linear-gradient(90deg, #3ce6ff, #4474ff)" };
    }

    if (pleasant.includes(emotion)) {
        return { level: "low", label: "Low", gradient: "linear-gradient(90deg, #3ce6ff, #4474ff)" };
    }

    return { level: "high", label: "High", gradient: "linear-gradient(90deg, #ff8d8d, #ff4a6f)" };
}

function getSupportedRecorderMimeType() {
    const candidates = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/ogg;codecs=opus",
        "audio/ogg",
        "audio/mp4"
    ];

    if (typeof MediaRecorder === "undefined" || !MediaRecorder.isTypeSupported) {
        return "";
    }

    for (const type of candidates) {
        if (MediaRecorder.isTypeSupported(type)) {
            return type;
        }
    }
    return "";
}

function extensionFromMime(mimeType) {
    const normalized = String(mimeType || "").toLowerCase();
    if (normalized.includes("ogg")) {
        return ".ogg";
    }
    if (normalized.includes("mp4")) {
        return ".mp4";
    }
    if (normalized.includes("webm")) {
        return ".webm";
    }
    if (normalized.includes("wav")) {
        return ".wav";
    }
    return ".webm";
}

async function convertBlobToWav(blob) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) {
        throw new Error("AudioContext is not supported in this browser.");
    }

    const audioContext = new AudioContextClass();
    try {
        const arrayBuffer = await blob.arrayBuffer();
        const decodedBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0));
        const wavArrayBuffer = audioBufferToWav(decodedBuffer);
        return new Blob([wavArrayBuffer], { type: "audio/wav" });
    } finally {
        try {
            await audioContext.close();
        } catch (closeError) {
            console.warn("AudioContext close warning:", closeError);
        }
    }
}

function audioBufferToWav(audioBuffer) {
    const numChannels = audioBuffer.numberOfChannels;
    const sampleRate = audioBuffer.sampleRate;
    const numSamples = audioBuffer.length;
    const bytesPerSample = 2;
    const blockAlign = numChannels * bytesPerSample;
    const byteRate = sampleRate * blockAlign;
    const dataSize = numSamples * blockAlign;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    writeAscii(view, 0, "RIFF");
    view.setUint32(4, 36 + dataSize, true);
    writeAscii(view, 8, "WAVE");
    writeAscii(view, 12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, 16, true);
    writeAscii(view, 36, "data");
    view.setUint32(40, dataSize, true);

    let offset = 44;
    for (let i = 0; i < numSamples; i += 1) {
        for (let channel = 0; channel < numChannels; channel += 1) {
            const sample = Math.max(-1, Math.min(1, audioBuffer.getChannelData(channel)[i]));
            const pcm = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
            view.setInt16(offset, pcm, true);
            offset += 2;
        }
    }

    return buffer;
}

function writeAscii(view, offset, text) {
    for (let i = 0; i < text.length; i += 1) {
        view.setUint8(offset + i, text.charCodeAt(i));
    }
}
function buildVocalInsight(stressLevel, emotion, confidencePct) {
    if (stressLevel === "high") {
        return `Elevated vocal tension detected with ${confidencePct}% confidence. The tone indicates strong arousal patterns linked to stress.`;
    }

    if (stressLevel === "low") {
        return `Stable vocal profile detected. ${emotion} cues are present with consistent rhythm and lower strain markers.`;
    }

    return `Moderate emotional fluctuation detected. Vocal energy suggests ${emotion.toLowerCase()} patterns with balanced stress markers.`;
}

function buildSuggestedAction(stressLevel, emotion) {
    if (stressLevel === "high") {
        return "Pause briefly, regulate breathing for one minute, and re-test in a quieter environment if possible.";
    }

    if (stressLevel === "low") {
        return `Current state appears regulated. Continue normal activity and monitor shifts if ${emotion.toLowerCase()} intensity changes.`;
    }

    return "Monitor your environment for stressors and maintain present awareness. A short reset break is recommended.";
}

if (dropZone && fileInput) {
    dropZone.addEventListener("click", () => fileInput.click());

    dropZone.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            fileInput.click();
        }
    });

    dropZone.addEventListener("dragover", (event) => {
        event.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (event) => {
        event.preventDefault();
        dropZone.classList.remove("dragover");
        if (event.dataTransfer.files.length > 0) {
            setSelectedFile(event.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (event) => {
        if (event.target.files.length > 0) {
            setSelectedFile(event.target.files[0]);
        }
    });
}

if (contactForm) {
    contactForm.addEventListener("submit", (event) => {
        event.preventDefault();
        contactForm.reset();
        alert("Thanks for reaching out. We will get back to you soon.");
    });
}

window.openTab = openTab;
window.toggleRecording = toggleRecording;
window.handleAnalyzeClick = handleAnalyzeClick;



