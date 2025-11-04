import { getAccessToken, logout } from './auth.js';

/**
 * A centralized API fetch helper.
 * - Adds Authorization header.
 * - Handles 401 errors by logging out.
 * - Throws an error for other non-ok responses.
 * @param {string} url - The API endpoint to call.
 * @param {object} options - The options for the fetch call (e.g., method, body).
 * @returns {Promise<any>} - The JSON response from the server.
 */
export async function apiFetch(url, options = {}) {
    const accessToken = getAccessToken();
    
    // Set default headers
    const defaultHeaders = {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${accessToken}`
    };

    // Merge headers
    const mergedOptions = {
        ...options,
        headers: {
            ...defaultHeaders,
            ...options.headers,
        }
    };

    // Make the request
    const response = await fetch(url, mergedOptions);

    // Handle 401 Unauthorized
    if (response.status === 401) {
        logout(); // Token is invalid or expired, log out
        return; // Stop further execution
    }

    // Handle other errors
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'An unknown API error occurred' }));
        throw new Error(errorData.detail || 'An API error occurred');
    }

    // Handle successful response
    // Check for empty body (e.g., 204 No Content)
    const contentType = response.headers.get("content-type");
    if (contentType && contentType.indexOf("application/json") !== -1) {
        return response.json();
    } else {
        return; // Return undefined for non-json responses
    }
}

