import * as dom from './dom.js';

/**
 * Adds a message to the chat box.
 * @param {string} text - The message content.
 * @param {string} sender - 'user' or 'bot'.
 * @param {string} [className=''] - Additional CSS class to add.
 */
export function addMessage(text, sender, className = '') {
    const message = document.createElement('div');
    message.classList.add('message', 'p-3', 'rounded-lg', 'max-w-[80%]', `${sender}-message`);
    if (className) {
        message.classList.add(className);
    }
    // Simple text escape
    const p = document.createElement('p');
    p.innerHTML = text; // Assumes text is safe or HTML-formatted feedback
    message.appendChild(p);
    dom.chatBox.appendChild(message);
    dom.chatBox.scrollTop = dom.chatBox.scrollHeight;
}

/**
 * Shows the typing indicator in the chat box.
 */
export function showTypingIndicator() {
    const indicator = document.createElement('div');
    indicator.id = 'typing-indicator';
    indicator.classList.add('message', 'bot-message', 'typing-indicator');
    indicator.innerHTML = `<span></span><span></span><span></span>`;
    dom.chatBox.appendChild(indicator);
    dom.chatBox.scrollTop = dom.chatBox.scrollHeight;
}

/**
 * Removes the typing indicator from the chat box.
 */
export function removeTypingIndicator() {
    const indicator = document.getElementById('typing-indicator');
    if (indicator) {
        indicator.remove();
    }
}

/**
 * Toggles the chat input and send button.
 * @param {boolean} disabled - Whether to disable the input.
 */
export function toggleInput(disabled) {
    dom.userInput.disabled = disabled;
    dom.sendBtn.disabled = disabled;
}

// --- View Switching ---

/**
 * Shows the dashboard view and hides others.
 */
export function showDashboardView() {
    dom.interviewContainer.classList.add('hidden');
    dom.resultsContainer.classList.add('hidden');
    dom.dashboardContainer.classList.remove('hidden');
}

/**
 * Shows the interview view and hides others.
 */
export function showInterviewView() {
    dom.dashboardContainer.classList.add('hidden');
    dom.resultsContainer.classList.add('hidden');
    dom.interviewContainer.classList.remove('hidden');
    dom.chatBox.innerHTML = ''; // Clear previous chat
}

/**
 * Shows the results view and hides others.
 */
export function showResultsView() {
    dom.interviewContainer.classList.add('hidden');
    dom.dashboardContainer.classList.add('hidden');
    dom.resultsContainer.classList.remove('hidden');
}

/**
 * Renders the final results on the results screen.
 * @param {object} data - The results data from the API.
 */
export function renderResults(data) {
    dom.finalSummaryCard.innerHTML = `<h3 class="text-2xl font-bold">${data.final_summary}</h3>`;
    
    dom.topicResultsContainer.innerHTML = '';
    data.all_topic_scores.forEach(topic => {
        const topicCard = document.createElement('div');
        topicCard.className = 'p-4 bg-gray-50 border border-gray-200 rounded-lg shadow-sm';
        topicCard.innerHTML = `
            <div class="flex justify-between items-center">
                <h4 class="text-xl font-semibold text-gray-700">${topic.topic}</h4>
                <span class="text-lg font-bold ${topic.final_score >= 7 ? 'text-green-600' : 'text-red-500'}">${topic.final_score} / 10</span>
            </div>
            <p class="mt-2 text-gray-600">${topic.summary}</p>
        `;
        dom.topicResultsContainer.appendChild(topicCard);
    });
}

// --- Transcript Modal UI ---

/**
 * Opens the transcript modal.
 */
export function openTranscriptModal() {
    dom.transcriptModal.classList.remove('hidden');
}

/**
 * Closes the transcript modal.
 */
export function closeTranscriptModal() {
    dom.transcriptModal.classList.add('hidden');
}

/**
 * Renders the loading state into the transcript modal.
 */
export function renderTranscriptLoading() {
    dom.transcriptContent.innerHTML = '<p class="text-center text-gray-500">Loading transcript...</p>';
}

/**
 * Renders an error message into the transcript modal.
 * @param {Error} error - The error object.
 */
export function renderTranscriptError(error) {
    dom.transcriptContent.innerHTML = `<p class="text-red-500 p-4">Error loading transcript: ${error.message}</p>`;
}

/**
 * Renders the full transcript into the modal.
 * @param {Array<object>} transcript - The transcript data from the API.
 */
export function renderTranscript(transcript) {
    let html = '';
    let currentTopic = '';
    
    const escapeHTML = (str) => {
      if (!str) return '';
      return str.replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }

    transcript.forEach(item => {
        if (item.topic !== currentTopic) {
            currentTopic = item.topic;
            html += `<h4 class="text-xl font-semibold mt-6 mb-3 p-2 bg-gray-100 rounded-lg sticky top-0">${currentTopic}</h4>`;
        }
        
        html += `
            <div class="my-3 p-4 border rounded-lg shadow-sm">
                <p class="font-bold text-gray-800">
                    <span class="text-blue-600">(${item.question_type})</span> AI Question:
                </p>
                <p class="ml-4 mt-1 text-gray-700">${escapeHTML(item.question)}</p>
                
                <p class="font-bold text-gray-800 mt-3">User Answer:</p>
                <p class="ml-4 mt-1 ${!item.user_answer ? 'text-gray-400 italic' : 'text-gray-700'}">${escapeHTML(item.user_answer) || 'No answer provided'}</p>
                
                <div class="mt-3 p-3 bg-blue-50 border-l-4 border-blue-300 rounded-md">
                    <p class="font-semibold text-blue-800">AI Feedback:</p>
                    <p class="ml-4 mt-1 ${!item.feedback ? 'text-gray-400 italic' : 'text-blue-700'}">${escapeHTML(item.feedback) || 'N/A'}</p>
                    <p class="font-semibold text-blue-800 mt-2">Score: <span class="font-normal text-blue-700">${item.score !== null ? item.score : 'N/A'} / 10</span></p>
                </div>
            </div>
        `;
    });
    dom.transcriptContent.innerHTML = html;
}

