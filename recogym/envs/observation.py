class Observation:
    def __init__(self, context, sessions, click=None):
        self.current_context = context
        self.current_sessions = sessions
        self.click = click

    def context(self):
        return self.current_context

    def sessions(self):
        return self.current_sessions

    # Allow to observe clicks in case of rewards for conversion instead of clicks
    def click(self):
        return self.click
        
