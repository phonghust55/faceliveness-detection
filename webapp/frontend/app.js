// ============================================================
//  Frontend logic: lấy frame từ webcam → POST /predict → vẽ kq
// ============================================================
const API_URL = "http://localhost:8000/predict";
const CAPTURE_INTERVAL_MS = 250;   // ~4 FPS – đủ cho anti-spoofing
const JPEG_QUALITY = 0.7;

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const btn = document.getElementById("toggle");

const $ = (id) => document.getElementById(id);
let stream = null;
let timer = null;
let inflight = false;        // chống chồng request

async function startCamera() {
    stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
        audio: false,
    });
    video.srcObject = stream;
    await video.play();

    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;

    $("status").textContent = "Đang stream...";
    btn.textContent = "Tắt webcam";
    timer = setInterval(captureAndSend, CAPTURE_INTERVAL_MS);
}

function stopCamera() {
    clearInterval(timer);
    timer = null;
    if (stream) stream.getTracks().forEach((t) => t.stop());
    stream = null;
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    $("status").textContent = "Đã dừng";
    btn.textContent = "Bật webcam";
}

async function captureAndSend() {
    if (inflight || video.readyState < 2) return;
    inflight = true;

    // Tạo offscreen canvas để encode JPEG
    const off = document.createElement("canvas");
    off.width = video.videoWidth;
    off.height = video.videoHeight;
    off.getContext("2d").drawImage(video, 0, 0);

    off.toBlob(
        async (blob) => {
            try {
                const fd = new FormData();
                fd.append("file", blob, "frame.jpg");
                const res = await fetch(API_URL, { method: "POST", body: fd });
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                const data = await res.json();
                renderResult(data);
            } catch (err) {
                console.error(err);
                $("status").textContent = "Lỗi: " + err.message;
            } finally {
                inflight = false;
            }
        },
        "image/jpeg",
        JPEG_QUALITY,
    );
}

function renderResult(d) {
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    $("found").textContent = d.found_face ? "có" : "không tìm thấy";
    $("latency").textContent = d.inference_ms.toFixed(1) + " ms";
    $("score").textContent = d.score_real.toFixed(4);

    const tag = $("label");
    if (!d.found_face) {
        tag.textContent = "-"; tag.className = "label-tag none";
        $("confidence").textContent = "-";
        return;
    }

    tag.textContent = d.label.toUpperCase();
    tag.className = "label-tag " + d.label;
    $("confidence").textContent = (d.confidence * 100).toFixed(1) + "%";

    // Vẽ bounding box
    const [x, y, w, h] = d.bbox;
    ctx.lineWidth = 3;
    ctx.strokeStyle = d.label === "real" ? "#16a34a" : "#dc2626";
    ctx.strokeRect(x, y, w, h);

    // Nhãn nền
    const text = `${d.label} ${(d.confidence * 100).toFixed(0)}%`;
    ctx.font = "bold 18px sans-serif";
    const tw = ctx.measureText(text).width + 12;
    ctx.fillStyle = ctx.strokeStyle;
    ctx.fillRect(x, y - 26, tw, 24);
    ctx.fillStyle = "#fff";
    ctx.fillText(text, x + 6, y - 8);
}

btn.addEventListener("click", () => {
    if (stream) stopCamera();
    else startCamera().catch((err) => alert("Không truy cập được webcam: " + err.message));
});
