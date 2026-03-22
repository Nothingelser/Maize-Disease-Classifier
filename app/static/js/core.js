// Core JavaScript for Maize Disease Classifier
class MaizeDiseaseClassifier {
    constructor() {
        this.token = localStorage.getItem('token');
        this.currentFile = null;
        this.recommendationsMap = {
            'Blight': [
                'Inspect nearby plants for lesion spread and isolate the worst-affected rows.',
                'Plan fungicide intervention only after confirming severity and field conditions.',
                'Capture follow-up images after treatment to monitor response.'
            ],
            'Gray Leaf Spot': [
                'Review field humidity and canopy density because both can accelerate spread.',
                'Prioritize repeat imaging on plants with rectangular lesions between veins.',
                'Use resistant hybrids and rotation planning where pressure remains high.'
            ],
            'Healthy': [
                'No disease signal is dominant in this sample, but continue routine scouting.',
                'Keep collecting clean samples to maintain a strong comparison baseline.',
                'Document environmental conditions if healthy leaves are being compared across fields.'
            ],
            'Maize Rust': [
                'Check upper and lower leaf surfaces for pustule density before treatment decisions.',
                'Track neighboring plants to see whether the outbreak is localized or expanding.',
                'Consider resistant varieties and targeted spraying if pressure keeps increasing.'
            ]
        };
        this.syncAuthState();
        this.init();
    }
    
    init() {
        this.setupUpload();
        this.setupEventListeners();
        this.checkAuth();
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
        const endpoint = hasToken ? '/api/predict' : '/api/predict/public';
        
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
                response = await fetch('/api/predict/public', {
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

        if (resultsSection) {
            resultsSection.style.display = 'block';
        }

        if (timestampText) {
            timestampText.textContent = new Date().toLocaleTimeString();
        }

        const probContainer = document.getElementById('probabilities');
        if (probContainer) {
            probContainer.innerHTML = '';
            data.probabilities.forEach(prob => {
                const percentage = (prob.probability * 100).toFixed(1);
                probContainer.innerHTML += `
                    <div class="probability-item">
                        <div class="probability-label">
                            <span>${prob.class}</span>
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
            const items = this.recommendationsMap[data.prediction] || [
                'Review the image manually before taking field action.',
                'Record the result in the reports workspace for later comparison.'
            ];
            recommendations.innerHTML = `<ul class="recommendation-list">${items.map((item) => `<li>${item}</li>`).join('')}</ul>`;
        }

        if (metadata) {
            metadata.innerHTML = `
                <h3>Image metadata</h3>
                <p class="section-copy">Basic details captured during the current browser session.</p>
                <div class="info-list" style="margin-top: 18px;">
                    <div class="info-item"><span>Predicted class</span><strong>${data.prediction}</strong></div>
                    <div class="info-item"><span>Processing time</span><strong>${data.processing_time.toFixed(0)} ms</strong></div>
                    <div class="info-item"><span>Top confidence</span><strong>${(data.confidence * 100).toFixed(1)}%</strong></div>
                    <div class="info-item"><span>Probability entries</span><strong>${data.probabilities.length}</strong></div>
                </div>
            `;
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
        const classes = {
            'Blight': 'badge-blight',
            'Gray Leaf Spot': 'badge-gray-leaf-spot',
            'Healthy': 'badge-healthy',
            'Maize Rust': 'badge-maize-rust'
        };
        return classes[prediction] || 'badge-healthy';
    }
}

// Initialize when DOM ready
document.addEventListener('DOMContentLoaded', () => {
    window.classifier = new MaizeDiseaseClassifier();
});
