// Core JavaScript for CropGuard workspace
class CropDiseaseClassifier {
    constructor() {
        this.token = localStorage.getItem('token');
        this.currentFile = null;
        this.confidenceThreshold = 0.6;
        this.classNames = [];
        this.displayClassNames = [];
        this.recommendationsMap = {};
        this.activeCrop = document.body?.dataset?.activeCrop || '';
        this.syncAuthState();
        this.init();
    }
    
    init() {
        this.setupUpload();
        this.setupEventListeners();
        this.loadModelInfo();
        this.checkAuth();
    }

    async loadModelInfo() {
        try {
            const response = await fetch(this.getApiPath('/model/info'));
            if (!response.ok) return;

            const data = await response.json();
            this.classNames = Array.isArray(data.classes) ? data.classes : [];
            this.displayClassNames = Array.isArray(data.display_classes)
                ? data.display_classes
                : this.classNames.map((label) => this.formatLabel(label));
            if (typeof data.confidence_threshold === 'number') {
                this.confidenceThreshold = data.confidence_threshold;
            }

            this.renderDiseaseLibrary();
            this.updateClassCount();
            this.updateLandingStats(data);
            this.updateConfidenceThresholdDisplay();
        } catch (error) {
            // Model metadata is optional for page bootstrap.
        }
    }

    getApiPath(path) {
        const normalizedPath = path.startsWith('/') ? path : `/${path}`;
        if (this.activeCrop) {
            return `/api/${this.activeCrop}${normalizedPath}`;
        }
        return `/api${normalizedPath}`;
    }

    updateConfidenceThresholdDisplay() {
        const thresholdDisplay = document.getElementById('confidenceThresholdDisplay');
        if (thresholdDisplay) {
            thresholdDisplay.textContent = `${(this.confidenceThreshold * 100).toFixed(0)}%`;
        }
    }

    updateLandingStats(modelInfo) {
        // Update hero stats on landing page
        const diseaseClassCount = document.getElementById('diseaseClassCount');
        if (diseaseClassCount && this.classNames.length) {
            diseaseClassCount.textContent = String(this.classNames.length);
        }

        const featureSpace = document.getElementById('featureSpace');
        if (featureSpace && modelInfo.features) {
            featureSpace.textContent = String(modelInfo.features);
        }

        const modelAccuracy = document.getElementById('modelAccuracy');
        if (modelAccuracy && modelInfo.accuracy) {
            const accuracyPercent = (modelInfo.accuracy * 100).toFixed(1);
            modelAccuracy.textContent = accuracyPercent + '%';
        }
    }

    updateClassCount() {
        const classesCount = document.getElementById('classesCount');
        if (classesCount) {
            const count = this.displayClassNames.length || this.classNames.length;
            if (count) classesCount.textContent = String(count);
        }
    }

    renderDiseaseLibrary() {
        const grid = document.getElementById('diseaseLibraryGrid');
        if (!grid || this.displayClassNames.length === 0) return;

        grid.innerHTML = this.displayClassNames.map((displayLabel) => {
            return `
                <div class="disease-card">
                    <div class="icon-wrap"><i class="fas fa-seedling"></i></div>
                    <h3>${displayLabel}</h3>
                    <p class="muted">Model-recognized class available in the current training set.</p>
                </div>
            `;
        }).join('');
    }

    formatLabel(label) {
        return String(label || '').replace(/___/g, ' / ').replace(/_/g, ' ').trim();
    }

    inferCondition(rawLabel, displayLabel) {
        const merged = `${rawLabel || ''} ${displayLabel || ''}`.toLowerCase();
        if (merged.includes('healthy')) return 'healthy';
        if (merged.includes('rust')) return 'rust';
        if (merged.includes('blight')) return 'blight';
        if (merged.includes('spot') || merged.includes('mold') || merged.includes('lesion')) return 'spot';
        return 'other';
    }

    getRecommendations(rawLabel, displayLabel) {
        const condition = this.inferCondition(rawLabel, displayLabel);
        const byCondition = {
            healthy: [
                'No dominant disease signal detected; continue routine scouting on nearby plants.',
                'Capture periodic comparison images to monitor for early changes over time.',
                'Maintain balanced irrigation and nutrition to sustain plant resilience.'
            ],
            rust: [
                'Inspect both leaf surfaces to confirm rust pustule density and spread pattern.',
                'Track surrounding plants to determine whether the outbreak is localized or expanding.',
                'Use integrated management, including resistant varieties and targeted treatment when warranted.'
            ],
            blight: [
                'Inspect adjacent plants for lesion expansion and remove heavily affected tissue where practical.',
                'Review humidity, canopy density, and irrigation timing that may be accelerating spread.',
                'Follow local agronomy guidance before fungicide use and document follow-up image checks.'
            ],
            spot: [
                'Increase scouting frequency to validate lesion progression across the field block.',
                'Reduce leaf wetness duration where possible through spacing and irrigation scheduling.',
                'Use crop-specific integrated disease management guidance from local extension services.'
            ],
            other: [
                'Review this result with field context before taking intervention decisions.',
                'Capture additional clear images from multiple leaves for confidence checks.',
                'Escalate uncertain or severe cases to a local crop specialist.'
            ]
        };
        return byCondition[condition];
    }
    
    setupUpload() {
        const uploadZone = document.getElementById('uploadZone');
        const fileInput = document.getElementById('fileInput');
        const cameraInput = document.getElementById('cameraInput');
        const browseBtn = document.getElementById('browseBtn');
        const selectFileBtn = document.getElementById('selectFileBtn');
        const cameraBtn = document.getElementById('cameraBtn');
        
        if (uploadZone) {
            uploadZone.addEventListener('click', (event) => {
                if (event.target.closest('button')) {
                    return;
                }
                fileInput.click();
            });
            uploadZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadZone.classList.add('dragover');
            });
            uploadZone.addEventListener('dragleave', () => {
                uploadZone.classList.remove('dragover');
            });
            uploadZone.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadZone.classList.remove('dragover');
                const files = e.dataTransfer.files;
                if (files.length) this.handleFile(files[0]);
            });
        }
        
        if (fileInput) {
            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length) this.handleFile(e.target.files[0]);
            });
        }

        if (cameraInput) {
            cameraInput.addEventListener('change', (e) => {
                if (e.target.files.length) this.handleFile(e.target.files[0]);
            });
        }

        browseBtn?.addEventListener('click', (event) => {
            event.preventDefault();
            event.stopPropagation();
            fileInput?.click();
        });

        selectFileBtn?.addEventListener('click', (event) => {
            event.preventDefault();
            fileInput?.click();
        });

        cameraBtn?.addEventListener('click', (event) => {
            event.preventDefault();
            event.stopPropagation();
            if (cameraInput) {
                cameraInput.click();
            } else {
                fileInput?.click();
            }
        });
    }

    handleFiles(files) {
        if (files && files.length) {
            this.handleFile(files[0]);
        }
    }
    
    async handleFile(file) {
        if (!file.type.match('image.*')) {
            this.showNotification('Please select an image file', 'error');
            return;
        }
        
        if (file.size > 16 * 1024 * 1024) {
            this.showNotification('File size must be less than 16MB', 'error');
            return;
        }
        
        this.currentFile = file;

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            const preview = document.getElementById('imagePreview');
            if (preview) {
                preview.src = e.target.result;
                document.getElementById('previewArea').style.display = 'block';
                document.getElementById('uploadZone').style.display = 'none';
            }
        };
        reader.readAsDataURL(file);
        
        // Upload and predict
        await this.predict(file);
    }
    
    async predict(file) {
        this.showLoading(true);
        
        const formData = new FormData();
        formData.append('file', file);
        const hasToken = this.hasValidToken();
        const endpoint = hasToken ? this.getApiPath('/predict') : this.getApiPath('/predict/public');
        
        try {
            let response = await fetch(endpoint, {
                method: 'POST',
                headers: hasToken ? { 'Authorization': `Bearer ${this.token}` } : {},
                body: formData
            });

            if (hasToken && response.status === 401) {
                this.clearAuthState();
                this.checkAuth();
                this.showNotification('Your session expired. Continuing with a public prediction.', 'error');
                response = await fetch(this.getApiPath('/predict/public'), {
                    method: 'POST',
                    body: formData
                });
            }
            
            const data = await response.json();
            
            if (data.success) {
                this.displayResults(data);
                this.showNotification(
                    data.saved ? 'Prediction complete and saved to your history.' : 'Prediction complete. Sign in to save history and exports.',
                    'success'
                );
            } else {
                this.showNotification(data.error || 'Prediction failed', 'error');
            }
        } catch (error) {
            this.showNotification('Error: ' + error.message, 'error');
        } finally {
            this.showLoading(false);
        }
    }

    resetWorkspace() {
        this.currentFile = null;

        const previewArea = document.getElementById('previewArea');
        const uploadZone = document.getElementById('uploadZone');
        const resultsSection = document.getElementById('resultsSection');
        const fileInput = document.getElementById('fileInput');
        const cameraInput = document.getElementById('cameraInput');
        const imagePreview = document.getElementById('imagePreview');
        const probabilities = document.getElementById('probabilities');
        const recommendations = document.getElementById('recommendations');
        const metadata = document.getElementById('metadata');
        const predictionText = document.getElementById('predictionText');
        const confidenceText = document.getElementById('confidenceText');
        const confidenceBar = document.getElementById('confidenceBar');
        const timestampText = document.getElementById('timestampText');
        const confidenceStatus = document.getElementById('confidenceStatus');

        if (previewArea) previewArea.style.display = 'none';
        if (uploadZone) uploadZone.style.display = 'flex';
        if (resultsSection) resultsSection.style.display = 'none';
        if (fileInput) fileInput.value = '';
        if (cameraInput) cameraInput.value = '';
        if (imagePreview) imagePreview.removeAttribute('src');
        if (predictionText) {
            predictionText.textContent = 'Waiting';
            predictionText.className = 'badge-large badge-healthy';
        }
        if (confidenceText) confidenceText.textContent = '0%';
        if (confidenceBar) confidenceBar.style.width = '0%';
        if (timestampText) timestampText.textContent = 'Not yet run';
        if (confidenceStatus) {
            confidenceStatus.className = 'confidence-status';
            confidenceStatus.textContent = 'Confidence status will appear after analysis.';
        }
        if (probabilities) {
            probabilities.innerHTML = '<div class="metadata-placeholder">Probabilities will appear here after prediction.</div>';
        }
        if (recommendations) {
            recommendations.textContent = 'Recommendations will appear after analysis.';
        }
        if (metadata) {
            metadata.innerHTML = `
                <h3>Image metadata</h3>
                <p class="section-copy">Basic details captured during the current browser session.</p>
                <div class="metadata-placeholder" style="margin-top: 18px;">
                    Upload an image to inspect file metadata.
                </div>
            `;
        }
    }
    
    displayResults(data) {
        const predictionText = document.getElementById('predictionText');
        const confidenceText = document.getElementById('confidenceText');
        const confidenceBar = document.getElementById('confidenceBar');
        const resultsSection = document.getElementById('resultsSection');
        const timestampText = document.getElementById('timestampText');
        const recommendations = document.getElementById('recommendations');
        const metadata = document.getElementById('metadata');
        const confidenceStatus = document.getElementById('confidenceStatus');
        const threshold = typeof data.confidence_threshold === 'number'
            ? data.confidence_threshold
            : this.confidenceThreshold;

        this.confidenceThreshold = threshold;
        this.updateConfidenceThresholdDisplay();

        if (predictionText) {
            predictionText.textContent = data.prediction;
            predictionText.className = `badge-large ${this.getBadgeClass(data.prediction)}`;
        }

        if (confidenceText) {
            confidenceText.textContent = `${(data.confidence * 100).toFixed(1)}%`;
        }

        if (confidenceBar) {
            confidenceBar.style.width = `${data.confidence * 100}%`;
        }

        const lowConfidence = Boolean(data.is_low_confidence) || (data.confidence < threshold);
        if (confidenceStatus) {
            if (lowConfidence) {
                confidenceStatus.className = 'confidence-status confidence-status-warning';
                confidenceStatus.textContent = data.confidence_message
                    || `Low confidence (< ${(threshold * 100).toFixed(0)}%). Capture another image before acting.`;
            } else {
                confidenceStatus.className = 'confidence-status confidence-status-ok';
                confidenceStatus.textContent = `Confidence is above threshold (${(threshold * 100).toFixed(0)}%).`;
            }
        }

        if (resultsSection) {
            resultsSection.style.display = 'block';
        }

        if (timestampText) {
            timestampText.textContent = new Date().toLocaleTimeString();
        }

        const probContainer = document.getElementById('probabilities');
        if (probContainer) {
            probContainer.innerHTML = '';
            const probabilities = Array.isArray(data.probabilities)
                ? [...data.probabilities].sort((a, b) => b.probability - a.probability)
                : [];
            probabilities.slice(0, 4).forEach(prob => {
                const percentage = (prob.probability * 100).toFixed(1);
                probContainer.innerHTML += `
                    <div class="probability-item">
                        <div class="probability-label">
                            <span>${prob.display_class || this.formatLabel(prob.class)}</span>
                            <span>${percentage}%</span>
                        </div>
                        <div class="progress">
                            <div class="progress-bar" style="width: ${prob.probability * 100}%; background: ${prob.color}"></div>
                        </div>
                    </div>
                `;
            });
        }

        if (recommendations) {
            const items = this.getRecommendations(data.prediction_label, data.prediction);
            recommendations.innerHTML = `<ul class="recommendation-list">${items.map((item) => `<li>${item}</li>`).join('')}</ul>`;
        }

        if (metadata) {
            metadata.innerHTML = `
                <h3>Image metadata</h3>
                <p class="section-copy">Basic details captured during the current browser session.</p>
                <div class="info-list" style="margin-top: 18px;">
                    <div class="info-item"><span>Crop app</span><strong>${data.crop ? this.formatLabel(data.crop) : 'All crops'}</strong></div>
                    <div class="info-item"><span>Predicted class</span><strong>${data.prediction}</strong></div>
                    <div class="info-item"><span>Processing time</span><strong>${data.processing_time.toFixed(0)} ms</strong></div>
                    <div class="info-item"><span>Top confidence</span><strong>${(data.confidence * 100).toFixed(1)}%</strong></div>
                    <div class="info-item"><span>Confidence threshold</span><strong>${(threshold * 100).toFixed(0)}%</strong></div>
                    <div class="info-item"><span>Probability entries</span><strong>${data.probabilities.length}</strong></div>
                </div>
            `;
        }

        if (lowConfidence && Array.isArray(data.top_predictions) && data.top_predictions.length > 1) {
            const secondChoice = data.top_predictions[1];
            this.showNotification(
                `Low-confidence prediction. Alternate class: ${secondChoice.display_class} (${(secondChoice.probability * 100).toFixed(1)}%).`,
                'error'
            );
        }
    }
    
    showLoading(show) {
        const spinner = document.getElementById('loadingOverlay');
        if (spinner) spinner.style.display = show ? 'flex' : 'none';
    }
    
    showNotification(message, type) {
        const container = document.querySelector('.toast-container');
        if (!container) {
            alert(message);
            return;
        }

        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;
        container.appendChild(toast);

        requestAnimationFrame(() => toast.classList.add('show'));
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 220);
        }, 2800);
    }

    syncAuthState() {
        if (!this.hasValidToken()) {
            this.clearAuthState();
        }
    }

    hasValidToken() {
        if (!this.token) {
            return false;
        }

        try {
            const payload = this.parseJwtPayload(this.token);
            if (!payload || !payload.exp) {
                return true;
            }

            return (payload.exp * 1000) > Date.now();
        } catch (error) {
            return false;
        }
    }

    parseJwtPayload(token) {
        const parts = token.split('.');
        if (parts.length !== 3) {
            throw new Error('Invalid JWT format');
        }

        const base64 = parts[1].replace(/-/g, '+').replace(/_/g, '/');
        const padded = base64.padEnd(Math.ceil(base64.length / 4) * 4, '=');
        return JSON.parse(atob(padded));
    }

    clearAuthState() {
        this.token = null;
        localStorage.removeItem('token');
        localStorage.removeItem('user');
    }
    
    checkAuth() {
        const authLinks = document.querySelectorAll('[data-auth-link]');
        const guestLinks = document.querySelectorAll('[data-guest-link]');
        const profileLinks = document.querySelectorAll('[data-profile-link]');

        authLinks.forEach((item) => {
            item.style.display = this.token ? 'none' : '';
        });

        guestLinks.forEach((item) => {
            item.style.display = this.token ? 'none' : '';
        });

        profileLinks.forEach((item) => {
            item.style.display = this.token ? '' : 'none';
        });
    }
    
    setupEventListeners() {
        const cropAppSelect = document.getElementById('cropAppSelect');
        cropAppSelect?.addEventListener('change', (event) => {
            const nextCrop = String(event.target.value || '').trim();
            if (!nextCrop) {
                window.location.href = '/workspace';
                return;
            }
            window.location.href = `/apps/${nextCrop}`;
        });

        // Logout button
        document.getElementById('logoutBtn')?.addEventListener('click', () => {
            this.clearAuthState();
            window.location.href = '/login';
        });
        
        document.getElementById('removeBtn')?.addEventListener('click', () => {
            this.resetWorkspace();
        });
    }

    getBadgeClass(prediction) {
        const normalized = String(prediction || '').toLowerCase();
        if (normalized.includes('healthy')) return 'badge-healthy';
        if (normalized.includes('blight')) return 'badge-blight';
        if (normalized.includes('rust')) return 'badge-maize-rust';
        if (normalized.includes('spot') || normalized.includes('mold') || normalized.includes('lesion')) {
            return 'badge-gray-leaf-spot';
        }
        return 'badge-generic';
    }
}

// Initialize when DOM ready
document.addEventListener('DOMContentLoaded', () => {
    window.classifier = new CropDiseaseClassifier();
});
