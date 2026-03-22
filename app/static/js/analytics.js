// Analytics Dashboard JavaScript
class AnalyticsDashboard {
    constructor() {
        this.charts = {};
        this.init();
    }
    
    init() {
        this.loadData();
        this.setupEventListeners();
    }
    
    async loadData() {
        const response = await fetch('/api/analytics?period=week', {
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
        });
        const data = await response.json();
        this.updateCharts(data);
        this.updateStats(data);
    }
    
    updateCharts(data) {
        // Update disease distribution chart
        const ctx = document.getElementById('diseaseChart');
        if (ctx && data.disease_distribution) {
            if (this.charts.disease) this.charts.disease.destroy();
            this.charts.disease = new Chart(ctx, {
                type: 'pie',
                data: {
                    labels: Object.keys(data.disease_distribution),
                    datasets: [{
                        data: Object.values(data.disease_distribution),
                        backgroundColor: ['#dc3545', '#ffc107', '#28a745', '#17a2b8']
                    }]
                }
            });
        }
    }
    
    updateStats(data) {
        document.getElementById('totalPredictions').textContent = data.total_predictions || 0;
        document.getElementById('avgConfidence').textContent = 
            data.average_confidence ? (data.average_confidence * 100).toFixed(1) + '%' : '0%';
        document.getElementById('mostCommon').textContent = data.most_common_disease || 'N/A';
    }
    
    setupEventListeners() {
        document.getElementById('periodSelect')?.addEventListener('change', (e) => {
            this.loadData(e.target.value);
        });
    }
}

// Initialize when DOM ready
document.addEventListener('DOMContentLoaded', () => {
    window.analytics = new AnalyticsDashboard();
});