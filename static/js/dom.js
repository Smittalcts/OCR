/**
 * This module selects and exports all necessary DOM elements
 * to avoid repeated getElementById calls.
 */

// Main Containers
export const dashboardContainer = document.getElementById('dashboard-container');
export const interviewContainer = document.getElementById('interview-container');
export const resultsContainer = document.getElementById('results-container');

// Dashboard Elements
export const dashboardContent = document.getElementById('dashboard-content');
export const userGreeting = document.getElementById('user-greeting');
export const logoutBtn = document.getElementById('logout-btn');

// Interview Chat Elements
export const chatBox = document.getElementById('chat-box');
export const userInput = document.getElementById('user-input');
export const sendBtn = document.getElementById('send-btn');
export const endBtn = document.getElementById('end-btn');

// Results Elements
export const finalSummaryCard = document.getElementById('final-summary-card');
export const topicResultsContainer = document.getElementById('topic-results-container');
export const backToDashboardBtn = document.getElementById('back-to-dashboard-btn');

// Transcript Modal Elements
export const transcriptModal = document.getElementById('transcript-modal');
export const transcriptContent = document.getElementById('transcript-content');

