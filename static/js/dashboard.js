import { apiFetch } from './api.js';
import * as dom from './dom.js';
import { getCurrentUser } from './state.js';

/**
 * Loads the correct dashboard based on the current user's role.
 */
export async function loadDashboard() {
    const user = getCurrentUser();
    if (!user) return;

    if (user.role === 'manager') {
        await loadManagerDashboard();
    } else {
        await loadAssociateDashboard();
    }
}

/**
 * Fetches data and renders the Manager dashboard.
 */
export async function loadManagerDashboard() {
    let contentHtml = `
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div class="bg-white p-6 rounded-lg shadow">
                <h3 class="text-xl font-semibold mb-4 text-gray-800">Schedule New Review</h3>
                <form id="schedule-form" class="space-y-4">
                    <div>
                        <label for="associate-select" class="block text-sm font-medium text-gray-700">Select Associate</label>
                        <select id="associate-select" name="associate_id" class="mt-1 block w-full p-2 border border-gray-300 rounded-md shadow-sm bg-white"></select>
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700">Select Topics</label>
                        <div class="mt-2 space-y-2">
                            <label class="flex items-center"><input type="checkbox" name="topics" value="Incident Management" class="mr-2" checked> Incident Management</label>
                            <label class="flex items-center"><input type="checkbox" name="topics" value="Problem Management" class="mr-2" checked> Problem Management</label>
                            <label class="flex items-center"><input type="checkbox" name="topics" value="Change Management" class="mr-2" checked> Change Management</label>
                        </div>
                    </div>
                    <button type="submit" class="w-full px-4 py-2 bg-green-500 text-white font-semibold rounded-lg shadow hover:bg-green-600 transition">Schedule</button>
                </form>
            </div>
            <div class="bg-white p-6 rounded-lg shadow md:col-span-2">
                <h3 class="text-xl font-semibold mb-4 text-gray-800">Scheduled Interview Status</h3>
                <div class="overflow-x-auto">
                    <table class="min-w-full divide-y divide-gray-200">
                        <thead class="bg-gray-50">
                            <tr>
                                <th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Associate</th>
                                <th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                                <th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Score</th>
                                <th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Scheduled On</th>
                                <th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Transcript</th>
                            </tr>
                        </thead>
                        <tbody id="results-table-body" class="bg-white divide-y divide-gray-200"></tbody>
                    </table>
                </div>
            </div>
        </div>`;
    dom.dashboardContent.innerHTML = contentHtml;
    
    // Fetch data and populate
    const data = await apiFetch('/api/manager/dashboard');
    
    // Populate associate dropdown
    const associateSelect = document.getElementById('associate-select');
    let associateOptions = '<option value="">Select an associate...</option>';
    data.associates.forEach(a => {
        associateOptions += `<option value="${a.id}">${a.username}</option>`;
    });
    associateSelect.innerHTML = associateOptions;

    // Populate results table
    const resultsTable = document.getElementById('results-table-body');
    let tableHtml = '';
    data.interview_results.forEach(r => {
        tableHtml += `
            <tr>
                <td class="px-4 py-2 whitespace-nowrap">${r.associate_username}</td>
                <td class="px-4 py-2 whitespace-nowrap">
                    <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                        r.status === 'completed' ? 'bg-green-100 text-green-800' :
                        r.status === 'pending' ? 'bg-yellow-100 text-yellow-800' :
                        'bg-gray-100 text-gray-800'
                    }">${r.status}</span>
                </td>
                <td class="px-4 py-2 whitespace-nowrap">${r.final_score !== null ? `${r.final_score} / ${r.max_score}` : 'N/A'}</td>
                <td class="px-4 py-2 whitespace-nowrap">${new Date(r.created_at).toLocaleDateString()}</td>
                <td class="px-4 py-2 whitespace-nowrap">
                    ${r.status === 'completed' ? `<button data-interview-id="${r.id}" class="view-transcript-btn px-3 py-1 bg-blue-100 text-blue-700 text-sm font-semibold rounded-md hover:bg-blue-200 transition">View</button>` : 'N/A'}
                </td>
            </tr>`;
    });
    resultsTable.innerHTML = tableHtml;
}

/**
 * Fetches data and renders the Associate dashboard.
 */
export async function loadAssociateDashboard() {
    const data = await apiFetch('/api/associate/dashboard');
    let contentHtml = `
        <div class="space-y-6">
            <!-- NEW: Practice Review Section -->
            <div class="bg-white p-6 rounded-lg shadow">
                <h3 class="text-xl font-semibold mb-4 text-gray-800">Practice Review</h3>
                <p class="text-gray-600 mb-4">Start an instant review for yourself. This will be saved to your "Completed Reviews" list.</p>
                <form id="practice-form" class="space-y-4">
                     <div>
                        <label class="block text-sm font-medium text-gray-700">Select Topics (at least one):</label>
                        <div class="mt-2 grid grid-cols-1 md:grid-cols-3 gap-2">
                            <label class="flex items-center p-2 bg-gray-50 rounded-md border hover:bg-gray-100 cursor-pointer"><input type="checkbox" name="topics" value="Incident Management" class="mr-2" checked> Incident Management</label>
                            <label class="flex items-center p-2 bg-gray-50 rounded-md border hover:bg-gray-100 cursor-pointer"><input type="checkbox" name="topics" value="Problem Management" class="mr-2" checked> Problem Management</label>
                            <label class="flex items-center p-2 bg-gray-50 rounded-md border hover:bg-gray-100 cursor-pointer"><input type="checkbox" name="topics" value="Change Management" class="mr-2" checked> Change Management</label>
                        </div>
                    </div>
                    <button type="submit" id="start-practice-btn" class="w-full px-4 py-2 bg-green-500 text-white font-semibold rounded-lg shadow hover:bg-green-600 transition">Start Practice</button>
                </form>
            </div>

            <!-- Existing Sections -->
            <div class="bg-white p-6 rounded-lg shadow">
                <h3 class="text-xl font-semibold mb-4 text-gray-800">Pending Reviews</h3>
                <div id="pending-interviews" class="space-y-3">
                    ${data.pending.length === 0 ? '<p class="text-gray-500">No pending reviews.</p>' : ''}
                </div>
            </div>
            <div class="bg-white p-6 rounded-lg shadow">
                <h3 class="text-xl font-semibold mb-4 text-gray-800">Completed Reviews</h3>
                <div id="completed-interviews" class="space-y-3">
                    ${data.completed.length === 0 ? '<p class="text-gray-500">No completed reviews.</p>' : ''}
                </div>
            </div>
        </div>`;
    dom.dashboardContent.innerHTML = contentHtml;

    // Populate pending
    let pendingHtml = '';
    data.pending.forEach(p => {
        pendingHtml += `
            <div class="flex justify-between items-center p-3 bg-gray-50 rounded-md border">
                <div>
                    <p class="font-medium">${JSON.parse(p.topics).join(', ')}</p>
                    <p class="text-sm text-gray-500">Scheduled: ${new Date(p.created_at).toLocaleString()}</p>
                </div>
                <button data-interview-id="${p.id}" class="start-interview-btn px-4 py-2 bg-blue-500 text-white font-semibold rounded-lg shadow hover:bg-blue-600 transition">Start</button>
            </div>`;
    });
    document.getElementById('pending-interviews').innerHTML = pendingHtml || '<p class="text-gray-500">No pending reviews.</p>';
    
    // Populate completed
    let completedHtml = '';
    data.completed.forEach(c => {
        completedHtml += `
            <div class="flex justify-between items-center p-3 bg-gray-50 rounded-md border">
                <div>
                    <p class="font-medium">${JSON.parse(c.topics).join(', ')}</p>
                    <p class="text-sm text-gray-500">${c.final_summary}</p>
                </div>
                <span class="text-lg font-bold ${c.final_score >= (c.max_score * 0.7) ? 'text-green-600' : 'text-red-500'}">${c.final_score} / ${c.max_score}</span>
            </div>`;
    });
    document.getElementById('completed-interviews').innerHTML = completedHtml || '<p class="text-gray-500">No completed reviews.</p>';
}

/**
 * Handles the schedule interview form submission.
 * @param {Event} e - The form submit event.
 */
export async function scheduleInterview(e) {
    e.preventDefault();
    const form = e.target;
    const formData = new FormData(form);
    const topics = Array.from(formData.getAll('topics'));
    const associate_id = formData.get('associate_id');
    
    if (!associate_id) {
        alert('Please select an associate.');
        return;
    }
    if (topics.length === 0) {
        alert('Please select at least one topic.');
        return;
    }

    try {
        await apiFetch('/api/manager/schedule', {
            method: 'POST',
            body: JSON.stringify({ associate_id: parseInt(associate_id), topics: topics })
        });
        alert('Interview scheduled!');
        loadManagerDashboard(); // Refresh dashboard
    } catch (error) {
        console.error('Failed to schedule interview:', error);
        alert('Error: ' + error.message);
    }
}

