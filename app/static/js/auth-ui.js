class AuthUI {
    constructor() {
        this.token = localStorage.getItem('token');
        this.user = this.readStoredUser();
    }

    readStoredUser() {
        try {
            const raw = localStorage.getItem('user');
            return raw ? JSON.parse(raw) : null;
        } catch (error) {
            return null;
        }
    }

    updateAdminLinks(isAdmin) {
        document.querySelectorAll('[data-admin-link]').forEach((item) => {
            item.style.display = isAdmin ? '' : 'none';
        });
    }

    async hydrateUser() {
        if (!this.token) {
            this.updateAdminLinks(false);
            return null;
        }

        if (this.user && typeof this.user.is_admin === 'boolean') {
            this.updateAdminLinks(this.user.is_admin);
            return this.user;
        }

        try {
            const response = await fetch('/api/user/profile', {
                headers: {
                    'Authorization': `Bearer ${this.token}`
                }
            });

            if (!response.ok) {
                this.updateAdminLinks(false);
                return null;
            }

            this.user = await response.json();
            localStorage.setItem('user', JSON.stringify(this.user));
            this.updateAdminLinks(Boolean(this.user.is_admin));
            return this.user;
        } catch (error) {
            this.updateAdminLinks(false);
            return null;
        }
    }
}

window.authUI = new AuthUI();
document.addEventListener('DOMContentLoaded', () => {
    window.authUI.hydrateUser();
});
