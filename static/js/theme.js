/**
 * Theme Toggle Functionality
 * Supports: light, dark, and system (follows OS preference)
 */

(function() {
    'use strict';

    const THEME_STORAGE_KEY = 'theme-preference';
    const THEMES = ['light', 'dark', 'system'];
    
    let currentTheme = 'system';
    let systemPreferenceQuery = null;
    let initialized = false;

    /**
     * Get system color scheme preference
     */
    function getSystemPreference() {
        if (window.matchMedia) {
            return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
        }
        return 'light';
    }

    /**
     * Apply theme to document
     */
    function applyTheme(theme) {
        const html = document.documentElement;
        
        if (theme === 'system') {
            const effectiveTheme = getSystemPreference();
            html.setAttribute('data-theme', effectiveTheme);
        } else {
            html.setAttribute('data-theme', theme);
        }
        
        currentTheme = theme;
        updateThemeButton();
    }

    /**
     * Update theme button icon and text
     */
    function updateThemeButton() {
        const buttons = document.querySelectorAll('.theme-toggle');
        buttons.forEach(button => {
            const icon = button.querySelector('i');
            const text = button.querySelector('span');
            
            if (icon) {
                // Remove all theme icons
                icon.classList.remove('fa-sun', 'fa-moon', 'fa-adjust', 'fa-desktop');
                
                // Add appropriate icon
                if (currentTheme === 'light') {
                    icon.classList.add('fa-sun');
                    if (text) text.textContent = 'Light';
                } else if (currentTheme === 'dark') {
                    icon.classList.add('fa-moon');
                    if (text) text.textContent = 'Dark';
                } else {
                    icon.classList.add('fa-adjust');
                    if (text) text.textContent = 'System';
                }
            }
        });
    }

    /**
     * Cycle through themes: light -> dark -> system -> light
     */
    function cycleTheme(event) {
        event.preventDefault();
        event.stopPropagation();
        
        const currentIndex = THEMES.indexOf(currentTheme);
        const nextIndex = (currentIndex + 1) % THEMES.length;
        const nextTheme = THEMES[nextIndex];
        
        setTheme(nextTheme);
    }

    /**
     * Set theme and save to localStorage
     */
    function setTheme(theme) {
        if (!THEMES.includes(theme)) {
            console.warn(`Invalid theme: ${theme}. Using 'system' instead.`);
            theme = 'system';
        }
        
        currentTheme = theme;
        try {
            localStorage.setItem(THEME_STORAGE_KEY, theme);
        } catch (e) {
            console.warn('Could not save theme preference:', e);
        }
        applyTheme(theme);
    }

    /**
     * Get saved theme preference or default to 'system'
     */
    function getSavedTheme() {
        try {
            const saved = localStorage.getItem(THEME_STORAGE_KEY);
            return saved && THEMES.includes(saved) ? saved : 'system';
        } catch (e) {
            console.warn('Could not read theme preference from localStorage:', e);
            return 'system';
        }
    }

    /**
     * Attach event listeners to theme toggle buttons
     */
    function attachEventListeners() {
        const buttons = document.querySelectorAll('.theme-toggle');
        buttons.forEach(button => {
            // Remove any existing listeners by cloning
            const newButton = button.cloneNode(true);
            button.parentNode.replaceChild(newButton, button);
            
            // Attach click listener
            newButton.addEventListener('click', cycleTheme, { passive: false });
        });
    }

    /**
     * Create theme toggle button if it doesn't exist
     */
    function createThemeButtonIfNeeded() {
        const existingButtons = document.querySelectorAll('.theme-toggle');
        if (existingButtons.length > 0) {
            return; // Buttons already exist
        }

        // Find navbars and add theme toggle button
        const navbars = document.querySelectorAll('.navbar');
        navbars.forEach(navbar => {
            const buttonContainer = navbar.querySelector('.d-flex.gap-2, .d-flex.gap-3, .d-flex:has(> a), .d-flex:has(> button)');
            
            if (buttonContainer) {
                const themeButton = document.createElement('button');
                themeButton.className = 'theme-toggle btn btn-outline-secondary btn-sm';
                themeButton.type = 'button';
                themeButton.setAttribute('aria-label', 'Toggle theme');
                themeButton.innerHTML = '<i class="fas me-2"></i><span></span>';
                
                // Insert at the beginning of button container
                const firstChild = buttonContainer.firstElementChild;
                if (firstChild) {
                    buttonContainer.insertBefore(themeButton, firstChild);
                } else {
                    buttonContainer.appendChild(themeButton);
                }
            }
        });
    }

    /**
     * Initialize theme system
     */
    function initTheme() {
        // Get saved preference or default to system
        const savedTheme = getSavedTheme();
        
        // Set up system preference listener
        if (window.matchMedia) {
            systemPreferenceQuery = window.matchMedia('(prefers-color-scheme: dark)');
            
            // Listen for system preference changes (only if in system mode)
            systemPreferenceQuery.addEventListener('change', function() {
                if (currentTheme === 'system') {
                    applyTheme('system');
                }
            });
        }
        
        // Apply theme
        setTheme(savedTheme);
    }

    /**
     * Initialize when DOM is ready
     */
    function init() {
        if (initialized) {
            return; // Prevent double initialization
        }
        
        function doInit() {
            // Initialize theme first
            initTheme();
            
            // Create buttons if needed
            createThemeButtonIfNeeded();
            
            // Attach event listeners
            attachEventListeners();
            
            // Update button appearance
            updateThemeButton();
            
            initialized = true;
        }
        
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', doInit);
        } else {
            doInit();
        }
    }

    // Expose functions globally for debugging/manual control
    window.setTheme = setTheme;
    window.getCurrentTheme = function() { return currentTheme; };

    // Initialize
    init();
})();
