import { apiFetch } from './api.js';
import { logout } from './auth.js';
import * as dom from './dom.js';
import * as state from './state.js';
import { loadDashboard, scheduleInterview } from './dashboard.js';
import { 
    sendMessage, 
    endInterview, 
    handleStartPractice, 
    startInterview, 
    loadAndShowTranscript 
} from './interview.js';
import { showDashboardView, closeTranscriptModal } from './ui.js';

/**
 * Main application initialization function.
 */
async function initializeApp() {
    if (!localStorage.getItem('accessToken')) {
        logout(); // No token, force logout
        return;
    }
    try {
        // Fetch current user details
        const user = await apiFetch('/api/users/me');
        if (!user) {
             logout(); // apiFetch might return undefined on 401, guard against it
             return;
        }
        state.setCurrentUser(user);
        
        // Set user greeting
        dom.userGreeting.textContent = `Welcome, ${user.username} (${user.role})`;
        
        // Load the appropriate dashboard
        loadDashboard();
        
        // Add all event listeners
        setupEventListeners();

    } catch (error) {
        console.error('Failed to load user data:', error);
        logout(); // Force logout on error
    }
}

/**
 * Centralized function to set up all global event listeners.
 */
function setupEventListeners() {
    // Auth
    dom.logoutBtn.addEventListener('click', logout);

    // Interview Chat
    dom.sendBtn.addEventListener('click', sendMessage);
    dom.endBtn.addEventListener('click', endInterview);
    dom.userInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && !dom.sendBtn.disabled) {
            sendMessage();
        }
    });

    // Results Screen
    dom.backToDashboardBtn.addEventListener('click', () => {
        showDashboardView();
        loadDashboard(); // Reload dashboard to show latest status
    });

    // Transcript Modal
    // Use event delegation for the close button in the modal
    dom.transcriptModal.addEventListener('click', (e) => {
        if (e.target.id === 'transcript-modal' || e.target.closest('#transcript-modal-close')) {
            closeTranscriptModal();
        }
    });

    // Event Delegation for dynamic content in dashboard
    dom.dashboardContent.addEventListener('submit', (e) => {
        if (e.target.id === 'schedule-form') {
            e.preventDefault();
            scheduleInterview(e);
        }
        if (e.target.id === 'practice-form') {
            e.preventDefault();
            handleStartPractice(e);
        }
    });

    dom.dashboardContent.addEventListener('click', (e) => {
        // Associate: Start scheduled interview
        const startBtn = e.target.closest('.start-interview-btn');
        if (startBtn) {
            e.preventDefault();
            const interviewId = startBtn.dataset.interviewId;
            startInterview(interviewId);
        }

        // Manager: View transcript
        const viewTranscriptBtn = e.target.closest('.view-transcript-btn');
        if (viewTranscriptBtn) {
            e.preventDefault();
            const interviewId = viewTranscriptBtn.dataset.interviewId;
            loadAndShowTranscript(interviewId);
        }
    });
}

// --- App Entry Point ---
// Run the initialization function when the DOM is loaded
document.addEventListener('DOMContentLoaded', initializeApp);

