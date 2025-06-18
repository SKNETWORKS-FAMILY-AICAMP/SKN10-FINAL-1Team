import uuid
import os
import sys
from typing import List, Dict, Any, Union

# --- Django Setup ---
# This section attempts to configure the Django environment if this module is used in a context
# where Django hasn't been set up yet (e.g., a standalone script or a separate process).
# Ideally, django.setup() should be called once at the application's main entry point.

# Add the 'backend' directory to sys.path to allow imports like 'conversations.models'.
# This assumes chat_history_service.py is at:
# d:\dev\SKN10-FINAL-1Team\backend\multiflow\src\multiflow\services\chat_history_service.py
# So, _project_root_django should point to d:\dev\SKN10-FINAL-1Team\backend
_current_file_dir = os.path.dirname(os.path.abspath(__file__))
_project_root_django = os.path.abspath(os.path.join(_current_file_dir, '..', '..', '..', '..'))

if _project_root_django not in sys.path:
    sys.path.insert(0, _project_root_django)

try:
    import django
    # Set DJANGO_SETTINGS_MODULE if not already set.
    # This typically points to your project's settings.py, e.g., 'config.settings'.
    # Adjust 'config.settings' if your Django settings file is located elsewhere relative to 'backend' dir.
    if not os.environ.get('DJANGO_SETTINGS_MODULE'):
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
    django.setup()
except Exception as e:
    print(
        f"WARNING: Initial Django setup in chat_history_service.py failed. "
        f"This service requires a properly configured Django environment to function. "
        f"Ensure DJANGO_SETTINGS_MODULE is correctly set and django.setup() is called, "
        f"preferably at your application's entry point. Error: {e}",
        file=sys.stderr
    )
    # Raising an ImportError as Django models won't be available.
    raise ImportError(
        f"Django setup failed, which is critical for DjangoChatHistory. "
        f"Original error: {e}"
    )

from asgiref.sync import sync_to_async
from conversations.models import ChatMessage, ChatSession, AgentType
from accounts.models import User
from crewai.memory.chat_history.base import BaseChatHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
# --- End Django Setup ---

class DjangoChatHistory(BaseChatHistory):
    """
    Manages chat history for a specific session using Django's ORM.
    Attempts to get or create a ChatSession instance upon initialization using a placeholder user.
    IMPORTANT: The user association is a placeholder and needs to be replaced with actual user context.
    """
    def __init__(self, session_id: uuid.UUID):
        super().__init__()
        self.session_id = session_id

        # TODO: Replace this with actual user and agent_type from the request context.
        # This is a temporary placeholder for development and testing.
        try:
            # Attempt to get the first user as a placeholder
            placeholder_user = User.objects.first()
            if not placeholder_user:
                print(f"WARNING: No users found in the database. ChatSession for {session_id} will not be created.")
                # Decide on behavior: raise error, or proceed without session persistence for history
                self.chat_session = None # Or handle as an error state
                return

            # Placeholder agent_type
            placeholder_agent_type = AgentType.AUTO 

            self.chat_session, created = ChatSession.objects.get_or_create(
                id=self.session_id,
                defaults={
                    'user': placeholder_user,
                    'agent_type': placeholder_agent_type,
                    # 'title': f"Session {session_id}" # Optional: set a default title
                }
            )
            if created:
                print(f"Created new ChatSession {self.session_id} for user {placeholder_user.email} with agent_type {placeholder_agent_type}")
            else:
                print(f"Retrieved existing ChatSession {self.session_id}")

        except Exception as e:
            print(f"ERROR: Could not get or create ChatSession {self.session_id}: {e}")
            # Depending on requirements, you might want to raise the error
            # or allow the history object to be created but non-functional for DB persistence.
            self.chat_session = None # Mark as non-functional for DB

    @sync_to_async
    def add_message(self, message: BaseMessage) -> None:
        if not self.chat_session:
            print(f"WARNING: ChatSession for {self.session_id} was not initialized. Message not added to DB.")
            return
        """
        Adds a message to the chat history for the current session.
        The message should be a dictionary with 'role' and 'content' keys.
        Example: {"role": "user", "content": "Hello"}
        """
        if not isinstance(message, BaseMessage):
            raise ValueError("Message must be an instance of langchain_core.messages.BaseMessage")

        role = ""
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, SystemMessage):
            role = "system"
        else:
            # Fallback for other BaseMessage types, or raise an error
            role = message.type # 'message.type' might be 'human', 'ai', 'system'
            # Consider raising ValueError for unsupported types if strict control is needed

        ChatMessage.objects.create(
            session_id=self.session_id,
            role=role,
            content=message.content
        )

    @sync_to_async
    def get_messages(self) -> List[BaseMessage]:
        if not self.chat_session:
            print(f"WARNING: ChatSession for {self.session_id} was not initialized. Returning empty message list.")
            return []
        """
        Retrieves all messages for the current session, ordered by creation time.
        Returns a list of dictionaries, each with 'role' and 'content'.
        """
        # Ensure messages are fetched for the correct session and ordered.
        messages_qs = ChatMessage.objects.filter(session_id=self.session_id).order_by('created_at')
        langchain_messages: List[BaseMessage] = []
        for msg in messages_qs:
            if msg.role == "user":
                langchain_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                langchain_messages.append(AIMessage(content=msg.content))
            elif msg.role == "system":
                langchain_messages.append(SystemMessage(content=msg.content))
            # Add other role mappings if necessary
        return langchain_messages

    @sync_to_async
    def clear(self) -> None:
        """
        Clears all messages from the chat history for the current session.
        """
        ChatMessage.objects.filter(session_id=self.session_id).delete()

# Example of how this class might be used (for illustration):
# async def main_example():
#     # This session_id should come from your application's session management logic
#     # and a ChatSession with this ID (and an associated user) must exist in the database.
#     example_session_id = uuid.uuid4()
#
#     # --- This part is CRITICAL and typically done elsewhere in your Django app --- 
#     # For this example to run standalone, you'd need a User and a ChatSession.
#     # from django.contrib.auth import get_user_model
#     # User = get_user_model()
#     # try:
#     #     user = await sync_to_async(User.objects.first)()
#     #     if not user:
#     #         user = await sync_to_async(User.objects.create_user)(username='testuser', password='password')
#     #     await sync_to_async(ChatSession.objects.get_or_create)(id=example_session_id, user=user)
#     # except Exception as e:
#     #     print(f"Error setting up example session/user: {e}")
#     #     return
#     # --- End critical setup --- 
#
#     history_manager = DjangoChatHistory(session_id=example_session_id)
#
#     print(f"Adding messages to session: {example_session_id}")
#     await history_manager.add_message({"role": "user", "content": "Hello, this is a test."})
#     await history_manager.add_message({"role": "assistant", "content": "Hi there, I am responding."})
#
#     retrieved_messages = await history_manager.get_messages()
#     print("Retrieved messages:")
#     for msg in retrieved_messages:
#         print(f"  {msg['role']}: {msg['content']}")
#
#     # print("Clearing messages...")
#     # await history_manager.clear()
#     # messages_after_clear = await history_manager.get_messages()
#     # print(f"Messages after clear: {len(messages_after_clear)}")

# if __name__ == '__main__':
#     # To run this example directly (requires Django environment to be fully set up):
#     # import asyncio
#     # asyncio.run(main_example())
#     pass
