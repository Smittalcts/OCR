/**
 * Gets the access token from local storage.
 * @returns {string | null} The access token.
 */
export const getAccessToken = () => localStorage.getItem('accessToken');

/**
 * Logs the user out by removing the token and redirecting to the login page.
 */
export const logout = () => {
    localStorage.removeItem('accessToken');
    window.location.href = '/'; // Redirect to login page
};

