(function () {
    const KEY = 'theme';
    const root = document.documentElement;

    function apply(theme) {
        root.setAttribute('data-theme', theme);
        document.querySelectorAll('.dark-mode-toggle i').forEach((icon) => {
            icon.className = theme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
        });
    }

    apply(localStorage.getItem(KEY) || 'light');

    document.addEventListener('click', (e) => {
        if (e.target.closest('.dark-mode-toggle')) {
            const next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
            localStorage.setItem(KEY, next);
            apply(next);
        }
    });
})();
