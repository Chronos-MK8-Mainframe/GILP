
"""
World State

Holds persistent objects like the Avatar.
Shared between Sandbox, Renderer, and Brain.
"""

from epsilon.env.avatar import Avatar

# Singleton Avatar Instance
avatar_instance = Avatar()

def update_world():
    avatar_instance.update()
