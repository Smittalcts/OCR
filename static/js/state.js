/**
 * This module holds the global client-side state for the application.
 */

let currentInterviewId = null;
let currentUser = null;

export const setInterviewId = (id) => {
    currentInterviewId = id;
};

export const setCurrentUser = (user) => {
    currentUser = user;
};

export const getInterviewId = () => currentInterviewId;
export const getCurrentUser = () => currentUser;

