// Dashboard JavaScript
class Dashboard {
    constructor() {
        this.token = localStorage.getItem('token');
        this.refreshTimer = null;
        this.selectedFeedbackId = null;
        this.feedbackRows = [];
        this.init();
    }
    
    async init() {
        const isAuthorized = await this.ensureAdminAccess();
        if (!isAuthorized) {
            return;
        }

        this.loadStats();
        this.loadFeedback();
        this.bindFeedbackEvents();
        this.setupRefresh();
    }

    async ensureAdminAccess() {
        if (!this.token) {
            window.location.href = '/login';
            return false;
        }

        try {
            const user = await window.authUI?.hydrateUser();
            if (!user || !user.is_admin) {
                window.location.href = '/analytics';
                return false;
            }

            return true;
        } catch (error) {
            window.location.href = '/analytics';
            return false;
        }
    }
    
    async loadStats() {
        try {
            const response = await fetch('/api/analytics/system', {
                headers: {
                    'Authorization': `Bearer ${this.token}`
                }
            });

            if (response.status === 401 || response.status === 403) {
                window.location.href = '/analytics';
                return;
            }

            const data = await response.json();
            this.updateStats(data);
        } catch (error) {
            console.error('Failed to load stats:', error);
        }
    }
    
    updateStats(data) {
        document.getElementById('totalUsers').textContent = data.users?.total || 0;
        document.getElementById('totalPredictions').textContent = data.predictions?.total || 0;
        document.getElementById('dailyAvg').textContent = data.predictions?.daily_avg?.toFixed(1) || 0;
        document.getElementById('errorsCount').textContent = data.errors_last_7_days || 0;
    }

    showToast(message, type = 'success') {
        const container = document.querySelector('.toast-container');
        if (!container) {
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
        }, 3200);
    }

    bindFeedbackEvents() {
        const statusFilter = document.getElementById('feedbackStatus');
        const refreshButton = document.getElementById('refreshFeedback');
        const closeModal = document.getElementById('closeReplyModal');
        const sendReplyBtn = document.getElementById('sendReplyBtn');

        statusFilter?.addEventListener('change', () => this.loadFeedback());
        refreshButton?.addEventListener('click', () => this.loadFeedback());
        closeModal?.addEventListener('click', () => this.closeReplyModal());
        sendReplyBtn?.addEventListener('click', () => this.submitReply());

        document.getElementById('feedbackBody')?.addEventListener('click', (event) => {
            const actionButton = event.target.closest('button[data-feedback-id]');
            if (!actionButton) {
                return;
            }

            const feedbackId = Number(actionButton.dataset.feedbackId);
            this.openReplyModal(feedbackId);
        });
    }

    async loadFeedback() {
        const status = document.getElementById('feedbackStatus')?.value || 'all';
        const body = document.getElementById('feedbackBody');
        if (!body) {
            return;
        }

        try {
            body.innerHTML = '<tr><td colspan="7" class="feedback-empty">Loading feedback...</td></tr>';
            const response = await fetch(`/api/admin/feedback?status=${encodeURIComponent(status)}&per_page=50`, {
                headers: {
                    'Authorization': `Bearer ${this.token}`
                }
            });

            if (response.status === 401 || response.status === 403) {
                window.location.href = '/analytics';
                return;
            }

            const data = await response.json();
            if (!response.ok) {
                this.showToast(data.error || 'Unable to load feedback', 'error');
                body.innerHTML = '<tr><td colspan="7" class="feedback-empty">Unable to load feedback.</td></tr>';
                return;
            }

            this.feedbackRows = data.feedback || [];
            this.renderFeedbackRows();
        } catch (error) {
            this.showToast(`Unable to load feedback: ${error.message}`, 'error');
            body.innerHTML = '<tr><td colspan="7" class="feedback-empty">Unable to load feedback.</td></tr>';
        }
    }

    renderFeedbackRows() {
        const body = document.getElementById('feedbackBody');
        if (!body) {
            return;
        }

        if (!this.feedbackRows.length) {
            body.innerHTML = '<tr><td colspan="7" class="feedback-empty">No feedback found for this filter.</td></tr>';
            return;
        }

        body.innerHTML = this.feedbackRows.map((item) => {
            const safeMessage = (item.message || '').replace(/</g, '&lt;').replace(/>/g, '&gt;');
            const createdAt = item.created_at ? new Date(item.created_at).toLocaleString() : '-';
            const contact = item.email || item.name || 'Not provided';
            const canReply = item.status !== 'replied';

            return `
                <tr>
                    <td>#${item.id}</td>
                    <td>${item.category || 'General'}</td>
                    <td>${contact}</td>
                    <td><span class="feedback-status status-${item.status || 'new'}">${item.status || 'new'}</span></td>
                    <td class="feedback-message-cell">${safeMessage}</td>
                    <td>${createdAt}</td>
                    <td>
                        ${canReply ? `<button class="btn btn-secondary btn-xs" data-feedback-id="${item.id}">Reply</button>` : '<span class="feedback-replied">Done</span>'}
                    </td>
                </tr>
            `;
        }).join('');
    }

    openReplyModal(feedbackId) {
        const modal = document.getElementById('feedbackReplyModal');
        const meta = document.getElementById('replyFeedbackMeta');
        const messageInput = document.getElementById('replyMessage');
        const row = this.feedbackRows.find((item) => Number(item.id) === Number(feedbackId));

        if (!modal || !messageInput || !row) {
            return;
        }

        this.selectedFeedbackId = feedbackId;
        meta.textContent = `Feedback #${row.id} | ${row.email || row.name || 'No contact email provided'}`;
        messageInput.value = '';
        modal.hidden = false;
    }

    closeReplyModal() {
        const modal = document.getElementById('feedbackReplyModal');
        const messageInput = document.getElementById('replyMessage');

        this.selectedFeedbackId = null;
        if (messageInput) {
            messageInput.value = '';
        }
        if (modal) {
            modal.hidden = true;
        }
    }

    async submitReply() {
        if (!this.selectedFeedbackId) {
            return;
        }

        const messageInput = document.getElementById('replyMessage');
        const reply = messageInput?.value.trim() || '';
        if (!reply) {
            this.showToast('Reply message is required', 'error');
            return;
        }

        try {
            const response = await fetch(`/api/admin/feedback/${this.selectedFeedbackId}/reply`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.token}`
                },
                body: JSON.stringify({ reply })
            });
            const data = await response.json();

            if (!response.ok) {
                this.showToast(data.error || 'Failed to send reply', 'error');
                return;
            }

            this.showToast(data.message || 'Reply sent');
            this.closeReplyModal();
            this.loadFeedback();
        } catch (error) {
            this.showToast(`Failed to send reply: ${error.message}`, 'error');
        }
    }
    
    setupRefresh() {
        this.refreshTimer = setInterval(() => {
            this.loadStats();
            this.loadFeedback();
        }, 30000);
    }
}

// Initialize when DOM ready
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new Dashboard();
});
