// Tab switching
function switchTab(tab, btn) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.getElementById(tab + '-tab').classList.add('active');
    btn.classList.add('active');
}

// Image preview
document.getElementById('image-input').addEventListener('change', function() {
    const file = this.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = e => {
            const preview = document.getElementById('image-preview');
            preview.src = e.target.result;
            preview.hidden = false;
            document.getElementById('image-btn').hidden = false;
            document.getElementById('image-result').hidden = true;
            document.getElementById('gradcam-box').hidden = true;
        };
        reader.readAsDataURL(file);
    }
});

// Video file selected
document.getElementById('video-input').addEventListener('change', function() {
    if (this.files[0]) {
        document.getElementById('video-btn').hidden = false;
        document.getElementById('video-result').hidden = true;
        document.getElementById('timeline').hidden = true;
    }
});

// Analyze image
async function analyzeImage() {
    const file = document.getElementById('image-input').files[0];
    if (!file) return;

    showLoading(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/predict/image', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        showLoading(false);
        displayImageResult(data);
    } catch (err) {
        showLoading(false);
        alert('Error: ' + err.message);
    }
}

function displayImageResult(data) {
    const box = document.getElementById('image-result');
    const isUnsafe = data.label === 'Unsafe';

    box.className = 'result-box ' + (isUnsafe ? 'unsafe' : 'safe');
    box.innerHTML = `
        <div class="result-label">
            ${isUnsafe ? '🔴' : '🟢'} ${data.label} — ${data.category}
        </div>
        <p>Confidence: <strong>${data.confidence}%</strong></p>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: ${data.confidence}%"></div>
        </div>
        <p style="color:#888; font-size:0.85rem;">
            Raw probability: ${data.raw_prob}
        </p>
    `;
    box.hidden = false;

    // Show Grad-CAM
    if (data.gradcam_url) {
        document.getElementById('gradcam-img').src = data.gradcam_url;
        document.getElementById('gradcam-box').hidden = false;
    }
}

// Analyze video
async function analyzeVideo() {
    const file = document.getElementById('video-input').files[0];
    if (!file) return;

    showLoading(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/predict/video', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        showLoading(false);
        displayVideoResult(data);
    } catch (err) {
        showLoading(false);
        alert('Error: ' + err.message);
    }
}

function displayVideoResult(data) {
    const box = document.getElementById('video-result');
    const isUnsafe = data.verdict === 'Unsafe';

    box.className = 'result-box ' + (isUnsafe ? 'unsafe' : 'safe');
    box.innerHTML = `
        <div class="result-label">
            ${isUnsafe ? '🔴' : '🟢'} Overall: ${data.verdict}
        </div>
        <p>Total frames analyzed: <strong>${data.total_frames_analyzed}</strong></p>
        <p>Safe frames: <strong style="color:#10b981">${data.safe_frames}</strong></p>
        <p>Unsafe frames: <strong style="color:#ef4444">${data.unsafe_frames}</strong></p>
    `;
    box.hidden = false;

    // Timeline
    const timeline = document.getElementById('timeline');
    timeline.innerHTML = '<h3 style="margin-bottom:10px">📊 Frame Timeline</h3>';
    data.timeline.forEach(item => {
        const div = document.createElement('div');
        div.className = 'timeline-item ' + item.label.toLowerCase();
        div.innerHTML = `
            <span>⏱ ${item.timestamp}s</span>
            <span>${item.label === 'Unsafe' ? '🔴' : '🟢'} ${item.label}</span>
            <span>${item.confidence}%</span>
        `;
        timeline.appendChild(div);
    });
    timeline.hidden = false;
}

function showLoading(show) {
    document.getElementById('loading').hidden = !show;
}