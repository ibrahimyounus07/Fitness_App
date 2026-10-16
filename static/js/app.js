// ─── GLOBAL STATE ────────────────────────────────────────────────────────────
const FitAI = {
    userId: localStorage.getItem('fitai_user_id'),
    userData: JSON.parse(localStorage.getItem('fitai_user_data')) || null,

    async init() {

        await this.loadUserData();

        this.setupSidebar();
        this.checkSetupRequired();

        if (this.userData && this.userData.role && this.userData.role.toLowerCase() === 'trainer') {

            if (window.location.pathname !== '/trainer') {
                window.location.href = '/trainer';
            }

        }

    },

    async loadUserData() {

        if (!this.userId) return;

        try {

            const res = await fetch(`/api/user/${this.userId}`);
            const data = await res.json();

            if (!data.error) {

                this.userData = data;

                localStorage.setItem(
                    'fitai_user_data',
                    JSON.stringify(data)
                );

                this.updateUI();

            }

        } catch (e) {

            console.error("Metabolic sync failed", e);

        }

    },

    updateUI() {

        if (!this.userData) return;

        const u = this.userData;
        const g = u.gamification || {};

        const nameEl = document.getElementById('sidebar-username');
        if (nameEl) nameEl.textContent = u.username || 'Neural User';

        const levelEl = document.getElementById('sidebar-level');
        if (levelEl) levelEl.textContent = `Tier ${g.level || 1}`;

        const xpEl = document.getElementById('top-xp');
        if (xpEl) xpEl.textContent = `${g.total_xp || 0} XP`;

        const streakEl = document.getElementById('streak-count');
        if (streakEl) streakEl.textContent = g.streak_days || 0;

        const trainerNav = document.getElementById('trainer-nav');

        if (trainerNav && u.role && u.role.toLowerCase() === 'trainer') {

            trainerNav.style.display = 'block';

            document.querySelectorAll('.sidebar-nav .nav-list li').forEach(li => {

                if (li.contains(trainerNav)) return;

                li.style.display = 'none';

            });

        }

    },

    checkSetupRequired() {

        const publicPaths = ['/', '/login'];

        if (!this.userId && !publicPaths.includes(window.location.pathname)) {

            window.location.href = '/';

        }

    },

    setupSidebar() {

        const toggle = document.getElementById('menu-toggle');
        const sidebar = document.getElementById('sidebar');
        const overlay = document.getElementById('sidebar-overlay');

        if (toggle && sidebar) {

            toggle.addEventListener('click', () => {

                sidebar.classList.toggle('open');

                if (overlay) overlay.classList.toggle('active');

            });

        }

        if (overlay) {

            overlay.addEventListener('click', () => {

                sidebar.classList.remove('open');
                overlay.classList.remove('active');

            });

        }

    }

};



// ─── PROFILE LOGIC ───────────────────────────────────────────────────────────
async function saveProfile() {

    const username = document.getElementById('setup-username')?.value?.trim();

    if (!username) {

        showToast('Profile Identifier Required', 'warning');
        return;

    }

    const password = document.getElementById('setup-password')?.value;

    if (!password) {

        showToast('Password Required for Neural Security', 'warning');
        return;

    }

    const role = document.getElementById('setup-role')?.value || 'user';

    const data = {

        username: username,
        password: password,
        role: role,
        email: document.getElementById('setup-email')?.value,
        fitness_goal: document.getElementById('setup-goal')?.value,
        experience_level: document.getElementById('setup-level')?.value,

        weight: 70,
        height: 175,
        target_weight: 65,
        activity_level: 'moderately active'

    };

    if (role === 'trainer') {

        delete data.email;
        delete data.fitness_goal;
        delete data.experience_level;
        delete data.weight;
        delete data.height;
        delete data.target_weight;
        delete data.activity_level;

    }

    const btn = document.getElementById('save-profile-btn');

    const originalHtml = btn.innerHTML;

    btn.innerHTML = '<i class="fas fa-circle-notch fa-spin"></i> Initializing...';

    btn.disabled = true;

    try {

        const res = await fetch('/api/user/setup', {

            method: 'POST',

            headers: { 'Content-Type': 'application/json' },

            body: JSON.stringify(data)

        });

        const result = await res.json();

        if (result.success) {

            localStorage.setItem('fitai_user_id', result.user_id);

            const userData = await fetch(`/api/user/${result.user_id}`)
                .then(r => r.json());

            localStorage.setItem(
                'fitai_user_data',
                JSON.stringify(userData)
            );

            showToast('Neural Profile Established! 🚀', 'success');

            setTimeout(() => {

                window.location.href = '/dashboard';

            }, 1200);

        } else {

            showToast(result.error || 'Interface Sync Failed', 'error');

            btn.innerHTML = originalHtml;
            btn.disabled = false;

        }

    } catch (e) {

        console.error('Save profile error:', e);

        showToast('Neural Network Timeout', 'error');

        btn.innerHTML = originalHtml;
        btn.disabled = false;

    }

}



// ─── UX COMPONENTS ───────────────────────────────────────────────────────────
function showToast(message, type = 'info') {

    let container = document.querySelector('.toast-container');

    if (!container) {

        container = document.createElement('div');

        container.className = 'toast-container';

        document.body.appendChild(container);

    }

    const icons = {

        success: '<i class="fas fa-check-circle"></i>',
        error: '<i class="fas fa-exclamation-triangle"></i>',
        info: '<i class="fas fa-info-circle"></i>',
        warning: '<i class="fas fa-bolt"></i>'

    };

    const toast = document.createElement('div');

    toast.className = `toast ${type} animate-fade-in`;

    toast.innerHTML = `<span>${icons[type] || icons.info}</span><span>${message}</span>`;

    container.appendChild(toast);

    setTimeout(() => {

        toast.style.opacity = '0';
        toast.style.transform = 'translateX(20px)';

        setTimeout(() => toast.remove(), 300);

    }, 4000);

}



// ─── LOGOUT HANDLING ─────────────────────────────────────────────────────────
function logout() {

    localStorage.removeItem('fitai_user_id');
    localStorage.removeItem('fitai_user_data');

    window.location.href = '/logout';

}



function attachLogoutListener() {

    const link = document.getElementById('logout-link');

    if (link) {

        link.addEventListener('click', (e) => {

            e.preventDefault();

            logout();

        });

    }

}



// ─── INITIALIZATION ──────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {

    FitAI.init();

    attachLogoutListener();

});