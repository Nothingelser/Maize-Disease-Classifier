// Export JavaScript
class ExportManager {
    constructor() {
        this.init();
    }
    
    init() {
        this.setupExportButtons();
    }
    
    setupExportButtons() {
        document.getElementById('exportPDF')?.addEventListener('click', () => this.exportPDF());
        document.getElementById('exportCSV')?.addEventListener('click', () => this.exportCSV());
        document.getElementById('exportExcel')?.addEventListener('click', () => this.exportExcel());
    }
    
    async exportPDF() {
        const predictionId = this.getCurrentPredictionId();
        if (!predictionId) return;
        
        window.open(`/api/export/${predictionId}?format=pdf`, '_blank');
    }
    
    async exportCSV() {
        const predictionIds = this.getSelectedPredictionIds();
        if (predictionIds.length === 0) {
            alert('Please select predictions to export');
            return;
        }
        
        const response = await fetch('/api/export/batch', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            },
            body: JSON.stringify({
                prediction_ids: predictionIds,
                format: 'csv'
            })
        });
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'predictions_export.csv';
        a.click();
        window.URL.revokeObjectURL(url);
    }
    
    async exportExcel() {
        const predictionIds = this.getSelectedPredictionIds();
        if (predictionIds.length === 0) {
            alert('Please select predictions to export');
            return;
        }
        
        const response = await fetch('/api/export/batch', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            },
            body: JSON.stringify({
                prediction_ids: predictionIds,
                format: 'excel'
            })
        });
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'predictions_export.xlsx';
        a.click();
        window.URL.revokeObjectURL(url);
    }
    
    getCurrentPredictionId() {
        return document.getElementById('currentPredictionId')?.value;
    }
    
    getSelectedPredictionIds() {
        const checkboxes = document.querySelectorAll('input[name="predictionIds"]:checked');
        return Array.from(checkboxes).map(cb => cb.value);
    }
}

// Initialize when DOM ready
document.addEventListener('DOMContentLoaded', () => {
    window.exportManager = new ExportManager();
});