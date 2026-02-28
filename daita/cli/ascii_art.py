"""
ASCII art utilities for Daita CLI.
"""

def get_daita_ascii_art():
    """Return the main DAITA ASCII art."""
    return """
██████╗  █████╗ ██╗████████╗ █████╗ 
██╔══██╗██╔══██╗██║╚══██╔══╝██╔══██╗
██║  ██║███████║██║   ██║   ███████║
██║  ██║██╔══██║██║   ██║   ██╔══██║
██████╔╝██║  ██║██║   ██║   ██║  ██║
╚═════╝ ╚═╝  ╚═╝╚═╝   ╚═╝   ╚═╝  ╚═╝

    AI Agent Framework & Platform
"""

def get_compact_daita_logo():
    """Return a compact DAITA logo for smaller displays."""
    return """
█▀▄ ▄▀█ █ ▀█▀ ▄▀█
█▄▀ █▀█ █  █  █▀█  Agents
"""

def display_welcome_banner():
    """Display the full welcome banner with ASCII art."""
    print(get_daita_ascii_art())

def display_compact_banner():
    """Display the compact banner."""
    print(get_compact_daita_logo())

def get_version_banner(version="0.1.0"):
    """Get version banner with ASCII art."""
    return f"""{get_daita_ascii_art()}
                 v{version}
     Build, test, and deploy AI agents
"""

def get_success_banner(message="Project created successfully!"):
    """Get success banner with ASCII art."""
    return f"""{get_daita_ascii_art()}
    ✨ {message}
"""