import { apiFetch } from './api.js';
import * as dom from './dom.js';
import * as state from './state.js';
import * as ui from './ui.js';
import { loadDashboard } from './dashboard.js';

/**
 * Starts a scheduled interview.
 * @param {string} interviewId - The ID of the interview to start.
 */
export async function startInterview(interviewId) {
    state.setInterviewId(interviewId);
    ui.showInterviewView();
    ui.addMessage("Initializing interview...", 'bot', 'status-message');
    
    try {
        const data = await apiFetch(`/api/interview/start/${interviewId}`, { 
            method: 'POST'
        });

        if (data.session_id && data.question) {
            state.setInterviewId(data.session_id); // Confirm ID
            ui.addMessage(data.question, 'bot');
            ui.toggleInput(false);
            dom.userInput.focus();
        } else {
            ui.addMessage('Error starting interview. Please check the server logs.', 'bot', 'error-message');
            setTimeout(() => {
                ui.showDashboardView();
                loadDashboard();
            }, 2000);
        }
    } catch (error) {
        ui.addMessage('Failed to connect. ' + error.message, 'bot', 'error-message');
        setTimeout(() => {
            ui.showDashboardView();
            loadDashboard();
        }, 2000);
    }
}

/**
 * Starts an ad-hoc practice interview.
 * @param {Event} e - The form submit event.
 */
export async function handleStartPractice(e) {
    e.preventDefault();
    const form = e.target;
    const formData = new FormData(form);
    const topics = Array.from(formData.getAll('topics'));

    if (topics.length === 0) {
        alert('Please select at least one topic to practice.');
        return;
    }

    const startBtn = dom.dashboardContent.querySelector('#start-practice-btn');
    if (startBtn) {
        startBtn.disabled = true;
        startBtn.textContent = 'Starting...';
    }

    try {
        const data = await apiFetch('/api/associate/start_practice', {
            method: 'POST',
            body: JSON.stringify({ topics: topics })
        });

        if (data.session_id && data.question) {
            state.setInterviewId(data.session_id);
            ui.showInterviewView();
            ui.addMessage("Practice session started. Good luck!", 'bot', 'status-message');
            ui.addMessage(data.question, 'bot');
            ui.toggleInput(false);
            dom.userInput.focus();
        } else {
            alert('Error starting practice. Please try again.');
        }
    } catch (error) {
        alert('Failed to start practice: ' + error.message);
    } finally {
        if (startBtn) {
            startBtn.disabled = false;
            startBtn.textContent = 'Start Practice';
        }
    }
}

/**
 * Sends the user's message to the backend.
 */
export async function sendMessage() {
    const messageText = dom.userInput.value.trim();
    const interviewId = state.getInterviewId();
    if (messageText === '' || !interviewId) return;

    ui.addMessage(messageText, 'user');
    dom.userInput.value = '';
    ui.toggleInput(true);
    ui.showTypingIndicator();

    try {
        const data = await apiFetch('/api/interview/submit', {
            method: 'POST',
            body: JSON.stringify({ interview_id: interviewId, answer: messageText })
        });
        
        ui.removeTypingIndicator();
        
        if (data.evaluation) {
            const evalText = `Feedback: ${data.evaluation.feedback} <br><b>[Answer Score: ${data.evaluation.score}/10]</b>`;
            ui.addMessage(evalText, 'bot', 'evaluation-message');
        }

        if (data.topic_score) {
            const topicText = `<strong>Topic Complete: ${data.topic_score.topic}</strong><br>Summary: ${data.topic_score.summary}<br><strong>Topic Score: ${data.topic_score.score}/10</strong>`;
            ui.addMessage(topicText, 'bot', 'topic-summary-message');
        }

        if (data.interview_finished) {
            ui.addMessage("You have completed all topics. Click below to see your final score.", 'bot');
            dom.endBtn.focus();
        } else if (data.next_question) {
            ui.addMessage(data.next_question, 'bot');
            ui.toggleInput(false);
            dom.userInput.focus();
        } else {
            ui.addMessage('The interview has concluded. Click below to see your final score.', 'bot', 'status-message');
            dom.endBtn.focus();
        }
    } catch (error) {
        ui.removeTypingIndicator();
        ui.addMessage('An error occurred. ' + error.message, 'bot', 'error-message');
        ui.toggleInput(false);
    }
}

/**
 * Ends the interview and displays the results.
 */
export async function endInterview() {
    const interviewId = state.getInterviewId();
    if (!interviewId) return;

    ui.toggleInput(true);
    ui.addMessage("Calculating final score...", 'bot', 'status-message');

    try {
        const data = await apiFetch('/api/interview/end', {
            method: 'POST',
            body: JSON.stringify({ interview_id: interviewId })
        });
        
        if (data.final_summary && data.all_topic_scores) {
            ui.showResultsView();
            ui.renderResults(data);
        } else {
            ui.addMessage('Could not retrieve final score.', 'bot', 'error-message');
        }
    } catch (error) {
        ui.addMessage('Error ending interview. ' + error.message, 'bot', 'error-message');
    } finally {
        state.setInterviewId(null); // Clear the interview ID
    }
}

/**
 * Fetches and displays the full interview transcript in a modal.
 * @param {string} interviewId - The ID of the interview to show.
 */
export async function loadAndShowTranscript(interviewId) {
    ui.openTranscriptModal();
    ui.renderTranscriptLoading();

    try {
        const transcript = await apiFetch(`/api/manager/interview_transcript/${interviewId}`);
        ui.renderTranscript(transcript);
    } catch (error) {
        ui.renderTranscriptError(error);
    }
}

